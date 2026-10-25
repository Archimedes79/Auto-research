# VS19 - DINO: look-forward-twice, mixed query selection, contrastive denoising

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S5, S4, T.*

## Papers

DINO (Zhang et al., ICLR 2023) https://arxiv.org/abs/2203.03605

## Idea

LOOK FORWARD TWICE: the box update of layer i is supervised through the NEXT layer's update applied to the undetached previous box, so the loss of layer i+1 also trains layer i. MIXED QUERY SELECTION: top-K encoder features initialise only the anchors; content queries stay learnable. CONTRASTIVE DENOISING: per GT box one positive query (small noise, reconstruct) and one negative (larger noise, predict no object).

## Formulation

Delta b_i = Layer_i(b_{i-1}); b_i' = Update(b_{i-1}, Delta b_i); b_i = Detach(b_i') is what is passed forward; the prediction of layer i is b_i^pred = Update(b_{i-1}', Delta b_i) with the UNDETACHED b_{i-1}' (Deformable DETR detaches in between). CDN: noise scales lambda1 < lambda2 (appendix lambda1 = 1.0, lambda2 = 2.0; the hyper-parameter table lists box noise scale 0.4, not reconciled); a group has 2n queries for n GT boxes; groups set dynamically (about 100 CDN pairs, 1000 DN queries for the large model); L1 + GIoU boxes, focal classes incl. negatives. 900 queries (300 x 3 patterns), 6 + 6 layers, mixed-query K not stated.

## Input

queries (point_set / boxes) + multi-scale rasters. output: boxes + classes.

## Training

hungarian_set + denoising.

## Differentiable?

partial (hungarian).

## Sensors and data

images (COCO).

## Reported results

COCO val ResNet-50, 12 epochs, AP: DN-DETR 43.4 -> optimised 44.9 -> + pure query selection 46.5 -> + mixed query selection 47.0 -> + look forward twice 47.4 -> + contrastive denoising 47.9; full DINO-4scale 49.0 (12 ep), 50.4 (24 ep), 50.9 (36 ep).

## Compute

not stated here.

## Code

URL from memory, not opened: https://github.com/IDEA-Research/DINO

## Fit to our project

reuse_idea - S - the look-forward-twice residual (undetached previous position + later delta in the loss) gives "no detach" credit assignment without sending gradients through the sampling positions; denoising is the DETR-side analogue of noisy teacher forcing.

## Needs

iterative per-stage position update.

## Conflicts

Hungarian matching for the main queries.

## Tried in our experiments

E39 cascade refiner small gain; E31 / E32 matched sparse decoder dropped.
