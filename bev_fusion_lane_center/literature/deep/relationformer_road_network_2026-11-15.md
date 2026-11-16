# Relationformer, road-network branch: deep read for the one-token-set design (E355)

Read 2026-11-15. **P** = paper https://arxiv.org/pdf/2203.10202, pages 1-19 seen as images (checked). **R** = code, branch `road_network_rgb` (20 US cities), https://raw.githubusercontent.com/suprosanna/relationformer/road_network_rgb/ ; **T** = branch `road_network` (Toulouse), same host. Code came through a summarising fetch tool: digest. Apache-2.0, 4 commits, no issue on token count or duplicates (https://github.com/suprosanna/relationformer). No lane, BEV or camera result exists. Field context: [scan](../scan_2026-11-15_set_based_lane_graph.md).

## 1. Architecture (US cities)
- ResNet-101; 4 encoder + 4 decoder layers of multi-scale deformable attention (4 levels, 4 points, 8 heads), width 512, FFN 1024 (R configs/road_rgb_2D.yaml; P p17).
- **80 [obj] tokens + 1 [rln] token** (R config; P p17). Tokens are `nn.Embedding(81, 2d)`; the **reference point is `Linear(pos).sigmoid()`**, a learned image-independent anchor per token (R models/deformable_detr_2D.py).
- [rln] is the last slot, joins decoder self-attention, **skips deformable cross-attention** (R). It never reads the image.
- **Relation head: last decoder layer only** (P p5-6; no intermediate outputs, no aux loss, R). Input `cat(obj_i, obj_j, rln)`, 3-layer MLP, 2 logits (R losses.py). Direction-free: random flip in training, both orders averaged at inference (R).
- Node head: 3-layer MLP, sigmoid; x, y used for L1 and matching, a fixed box size (0.2 P p8, p17; 0.15 R) only for GIoU.

## 2. Matching and loss
- Hungarian per image: cost = **5 x L1(x, y) + 3 x (-p_obj)**, GIoU cost 0 (P p17 Table 7; R models/matcher.py). Toulouse 5 / 2 / 0.
- Edges are **not matched**: relations are scored between tokens that node matching assigned to GT nodes; the GT adjacency is remapped to them (P p7; R losses.py). Unmatched tokens get no edge loss.
- Sampling: paper, 3 random background pairs per valid one (P p7). Code, R: at most 40 positives, filled to **80 pairs per graph** with random negatives; T: 20 / 40 (digest). No fixed ratio.
- Weights (reg / GIoU / cls / rln): 5 / 2 / 3 / 4 US, 5 / 2 / 2 / 1 Toulouse (P p17). Class CE 0.2 / 0.8; L1 divided by GT node count (R). The config adds a cardinality term (W_CARD 1.0) that Eq. 2 lacks (R config; P p7).

## 3. Tokens versus graph size
- N must exceed the maximum number of objects (P p4). **Not stated:** nodes per patch, node spacing in px or m, behaviour when nodes exceed tokens.
- Data: 180 images of 2048 x 2048 (144 / 9 / 27), overlapping 128 x 128 patches, 124k / 13k / 25k (P p7, p15). GT is only simplified: a degree-2 node goes when the angle exceeds 160 degrees (P p15; R generate_data.py). No resampling, no minimum distance, no node-count filter; only patches with < 2 nodes or < 1 edge are skipped (R). My inference: more than 80 nodes would leave GT unmatched and break the edge remap.
- Only over-provisioning evidence, on Visual Genome: tokens 75 / 100 / 200 / 300 give AP50 25.1 / 25.8 / 26.3 / 26.3 (P p18). Our plan, about 1600 tokens for about 110 kept nodes (E355 guess), is roughly 15 : 1; nothing here tests that.

## 4. Budget and numbers
- AdamW, lr 1e-4 (P p17) or 2e-4, backbone 2e-5, decay at epoch 80, clip 0.1 (R config; they disagree); batch 32, 100 epochs (P p17). GPUs, hours, FLOPs, road parameters: **not stated**. Visual Genome: 92.9 M, 18.2 FPS on a GTX 1080 (P p9).
- 20 US cities TOPO P / R / F1 85.28 / 77.75 / 81.34 vs Sat2Graph 80.70 / 72.28 / 76.26; node mAP 29.25, edge mAP 33.19; Toulouse F1 99.37 (P p8). Authors blame parallel roads for the weak node and edge scores.
- [rln] ablation: vessel edge mAP 76.3 to 78.1 (P p11 Table 5); Visual Genome MLP head AP 26.0 to 26.3 (P p11 Table 4).

## 5. Inference (R inference.py)
Node = argmax of 2 logits, **no score threshold**; optional class-agnostic NMS (IoU 0.90, near no-op). Pairs = **all unordered pairs** of kept nodes, both orders averaged, edge = argmax. No junction merge or spacing rule. The paper names no threshold (P p6).

## 6. Dense-initialised variant: what changes
- Tokens: replace `query_embed` and the `reference_points` linear by lattice cell centres plus sampled features. A `two_stage` branch (top-k encoder proposals as reference points) exists but is off in both configs (R deformable_detr_2D.py, digest): the nearest hook to DDQ.
- Self-attention over 1600 tokens becomes windowed; deformable cross-attention is linear and stays.
- Matcher: add type and heading cost or a local one-to-one rule; CE 0.2 / 0.8 is too weak at 1 : 15, use the unused focal loss (R losses.py).
- Losses: drop GIoU and cardinality; add keep / prune and type CE; the edge loss stays.
- Inference: all-pairs `combinations` is O(n^2): prune first, add a neighbour gate.

## 7. What to reuse for us
1. Pair head `cat(node_i, node_j, rln)`, 3-layer MLP; rln as one global query.
2. Direction-free pair training (random flip, average both orders).
3. Relation head after the last layer only = our connect 2 on survivors.
4. Edge loss on matched tokens with the adjacency remapped through the matching: copy this logic.
5. About 80 sampled pairs per graph; use hard negatives (near, unconnected) instead of random.
6. Node cost 5 x L1 + 3 x p_obj, no GIoU for points.
7. Degree-2 pruning at 160 degrees matches our Douglas-Peucker target; no spacing rule to borrow.
8. [rln] skips image cross-attention: cheap, keep.
9. Leave: deformable CUDA ops (compile step), all-pairs inference, CE 0.2 / 0.8, the `pdb.set_trace` in `loss_edges` (digest).
10. Gap: no duplicate or spacing study, so E355's duplicate kill rule has no precedent here.
