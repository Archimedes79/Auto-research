# can22-minimalcycles: method, losses, results

Status of facts: checked = read in the paper full text (ar5iv, https://ar5iv.labs.arxiv.org/abs/2112.10155) or in the code file named, through the page fetch tool (extracts, not a download). "our arithmetic" = computed by us from checked numbers.

## Representation (paper section 3, checked)
- Curve c: directed lane centreline, Bezier with 3 control points, coordinates normalised to [0,1]^2 (theta = 6 numbers). A curve is a whole centreline, not a short piece between two junction nodes.
- Successor / incidence matrix A in {0,1}^(N x N): A[x,y] = 1 if the start of y meets the end of x.
- Intersection: geometric crossing of two curves; I_c = ordered list of the crossings along curve c (at most one crossing per pair, no self crossing, no floating curves are assumed).
- Segment = part of a curve between two successive crossings. Minimal cycle = closed polycurve of segments whose inside no other curve crosses. Cover B = set of curves (and boundary segments) that carry the segments of one cycle. Paper: a cover identifies its cycle (Lemma 3.1), the set of covers identifies the crossing order (Corollary 3.2.1). Not re-derived here.

## Network, transformer variant (paper + src/detr/detr.py, checked)
- Input: image 448x800 (paper; config patch_size [800,448]), calibration matrix. Backbone: paper says DeepLabv3+ pretrained on Cityscapes; code backbone is `SPPNet` from `src/deeplab`, calibration enters only as BEV positional encoding (`bev_pos`), no explicit lift of features to BEV in that file.
- Decoder output `hs` [L, B, Q+P, d]; Q = 100 curve queries, P = 100 cycle queries (separate query sets for left / right traffic). Only the last layer is used.
- Curve heads on the first Q: class logits [B,Q,3] (2 classes + no object), control points [B,Q,6] (sigmoid), end points [B,Q,4], association features [B,Q,D_assoc] (MLP, ReLU, no output activation).
- Cycle heads on the last P: membership logits [B,P,Q+5] (Q curve slots + 5 fixed boundary segments: upper, left, right, bev_left, bev_right), existence [B,P,2], centre [B,P,2] (sigmoid).
- Pair head: for K selected queries, K x K grid of ordered pairs [f_i ; f_j] (2 D_assoc) -> MLP -> 1 logit (asymmetric; diagonal kept). Training: K = Hungarian-matched queries only. Inference: queries with class-1 probability > 0.5, sigmoid on the logit, threshold 0.5 (`assoc_thresh` in confusion.py). Widths and depths come from a `parameter_setting` dict not seen.
- Polygon-RNN variant (paper, not read in code): fully connected sub-network predicts a grid of initial points, Polygon-RNN produces control points, a transformer decoder then predicts cycles from the padded RNN states. Not needed for us.

## Losses (paper section 3.4 "Training Framework" and code, checked)
Paper: L = L_curve + alpha L_cycle; L_curve = L_splines + beta_e L_exists + beta_c L_connect; L_cycle = L_member + beta_d L_exists + beta_f L_center. Values of alpha, beta are not in the paper text.
Code (SetCriterion in detr.py, coefficients are argparse defaults of train_tr_nusc.py):
| term (key) | what | coefficient |
|---|---|---|
| loss_ce | cross entropy of curve class, no-object weight eos 0.3 | detection 3 |
| loss_polyline | Bezier matrix N->100 sampled points, pairwise L1 cdist, min in both directions, mean (Chamfer-like; paper text says L1 on control points) | 3 |
| loss_endpoints | L1 of end points (only if `single_frame`) | 1 |
| loss_assoc | BCE with logits on the K x K pair logits vs `con_matrix` re-indexed by matched ids; weight 3*lab + (1-lab) | 2 |
| loss_end_match | positives: 2 x L1 distance between end of i and start of j; negatives: 3 x max(0.05 - d, 0) | 1 |
| loss_poly_hamming | BCE with logits on cycle membership [P,Q+5] vs multi-hot target, weight 3*gt + (1-gt); plus loss_minimality (hinge: curves that are non-members must be farther from the cycle centre than the nearest member) | 4 and 5 |
| loss_poly_exist | cross entropy, negative weight 0.3 | 4 |
| loss_poly_center | L1 | 1 |
| loss_naive | pairwise proximity-based membership (not in the default loss list) | 0.5 |
The default loss list of `build()` is labels, boxes, loss_polyline, endpoints, assoc; the polygon losses are added inside the train loop of train_tr_nusc.py when the matcher reports no problem.

## What is differentiable (code, checked)
- Gradient: all loss terms above w.r.t. query embeddings and Bezier control points of the matched queries.
- No gradient: Hungarian matching (`@torch.no_grad`), the re-labelling of GT cycles to estimated curves (`min_max` mode of the matcher: nearest-GT assignment under a distance threshold 0.05, builds the [num_polygons, 105] target mask), the thresholds and the end-point merge at test time. Unmatched queries get no pair supervision.

## Supervision sources (code, checked on dataset.py and defaults.yml)
- Curve set: nuScenes `lane + lane_connector` tokens of the map API, label PNG per camera token with lane ids, Bezier fitted to the pixels of each id (roads under 10 pixels dropped).
- Pair target `con_matrix`: from `get_outgoing_lane_ids` of the map API (successors), not from end-point distance. `start_con_matrix` / `fin_con_matrix` (shared start / shared end, L1 below 0.01 normalised) are built but have loss weight 0.
- Cycle targets `poly_one_hots`, `poly_centers`: read from precomputed dicts (config paths `poly_dict_path` = GT-based, `poly_pretrain_poly_dict_path` = based on estimated curves). In dataset.py the polygon targets are placeholders; an unused raster routine (`get_polygons`, free-space blobs by distance transform) exists. The graph-based cycle extraction of the paper was not found in the files read.
- Order labels `gt_order_labels`: sampled-point proximity (below 0.01) between curves and boundaries, padded to 20.

## Schedule (train_tr_nusc.py and defaults.yml, checked; they disagree)
AdamW, lr 1e-5 for all groups, weight decay 1e-4, StepLR step 50, grad clip 0.1; batch size 1 (config), epochs 200 (config) or 300 (argparse default); BatchNorm frozen; only backbone blocks 18-21 and spp trainable; by default the script loads `final_ckpts/TR_MC_Nusc.pth` (a fine-tune from the released checkpoint) unless `load_pretrained_backbone` is set. README recommends a warm start from the pretrained STSU model; the paper text read does not mention it. BEV extent x -25..25 m, z 1..50 m, 0.25 m per cell. Split of Roddick and Cipolla (2020).

## Results (paper Tables 1-4, checked; ar5iv full text)
Metrics as named in the paper: M-F (map F-score), Detect (detection ratio), C-F (connectivity F-score), MC-F (minimal-cycle F), I-Order (normalised Levenshtein distance of intersection order, lower is better; unmatched GT curves count 2). Thresholds of the matching are not stated in the text.
nuScenes (Table 1): PINET 49.5 / 19.2 / - / 14.7 / 1.08; PRNN 52.9 / 40.5 / 24.5 / 45.9 / 0.894; Ours/PRNN 51.7 / 53.1 / 49.9 / 53.2 / 0.824; TR 56.7 / 59.9 / 55.2 / 62.0 / 0.800; TR-RNN 58.0 / 59.7 / 52.3 / 60.3 / 0.791 (RNN-order 0.939); Ours/TR 58.2 / 60.2 / 55.3 / 62.5 / 0.776 (columns M-F / Detect / C-F / MC-F / I-Order).
Argoverse (Table 2): PINET 47.2 / 15.1 / - / 24.5 / 1.23; PRNN 45.1 / 40.2 / 31.3 / 42.8 / 1.09; Ours/PRNN 44.3 / 55.4 / 46.4 / 49.5 / 1.05; TR 55.6 / 60.1 / 54.9 / 57.4 / 0.893; TR-RNN 55.8 / 53.6 / 54.4 / 52.2 / 0.953 (RNN-order 0.951); Ours/TR 57.1 / 64.2 / 58.1 / 58.5 / 0.883.
Table 3 (Polygon-RNN with GT initial points, nuScenes M-F / Detect / C-F / MC-F / I-Order): PRNN(GT) 71.1 / 76.4 / 52.9 / 66.9 / 0.645; Ours/PRNN(GT) 72.6 / 77.2 / 55.0 / 67.5 / 0.642. Argoverse: C-F 54.1 -> 54.5. Table 4 (cycle head, H-GT-F / H-EST-F): Ours/TR 60.9 / 73.0 nuScenes, 56.6 / 61.5 Argoverse. One read each.
Deltas, our arithmetic, transformer without -> with cycle branch: C-F +0.1 (nuScenes), +3.2 (Argoverse); MC-F +0.5, +1.1; Detect +0.3, +4.1; I-Order -0.024, -0.010.
Speed: 11 FPS without batching including association steps; GPU not stated. Parameter overhead of the cycle branch: "negligible" (paper).

## Not shown
Pair accuracy per junction or per pair type; ablation of positive weight 3, of loss_end_match, of the matched-only supervision; seeds, intervals; comparison with later lane-graph methods; any non-camera input; how well the cycle branch works with a better pair head (it is trained on cycles of the estimated curves, so errors of the curve head propagate into its targets).
