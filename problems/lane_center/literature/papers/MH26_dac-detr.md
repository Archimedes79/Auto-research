# MH26 - DAC-DETR (auxiliary decoder without self-attention, one-to-many by score threshold)

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M** (A abstract, M method, C code page). Verified: checked. Pipeline stage: L, T.*

## Papers

DAC-DETR: Divide the Attention Layers and Conquer (Hu, Sun, Wang, Yang, NeurIPS 2023) https://papers.nips.cc/paper_files/paper/2023/hash/edd0d433f8a1a51aa11237a6543fc280-Abstract-Conference.html (method read from a text copy of the paper at object.cloud.sdsc.edu)

## Idea

Cross-attention gathers several queries on one object, self-attention disperses them. A training-only auxiliary decoder without self-attention (shared cross-attention, FFN, queries, head) is trained one-to-many to strengthen the gathering; the main decoder keeps Hungarian one-to-one. The auxiliary decoder is dropped at inference.

## Formulation

per GT, score M = p_q(c_hat) + IoU(b_q, b_hat) over all queries; positive if M above a threshold t and among the top-k (t and k not given in the text read); L = L_O (one-to-one) + L_C (focal + GIoU); Hungarian with repeated GT in the auxiliary branch is worse (45.5 vs 47.1 AP). - input / output: DETR decoders; boxes.

## Training

COCO 12 epochs; regime `hungarian_set` (main) + `matching_free` (aux).

## Differentiable?

yes.

## Sensors and data

COCO.

## Reported results

Deformable-DETR R50 12 ep 43.7 -> 47.1 AP (+3.4); DINO 49.0 -> 50.0; with the Align-DETR loss 50.9 (R50) / 58.1 (Swin-L). Ablations: frozen cross-attention in main decoder 45.9; self-attention in the auxiliary decoder 43.1; one-to-one in the auxiliary decoder 43.5.

## Compute

overhead not reported.

## Code

https://github.com/huzhengdongcs/DAC-DETR ("will be released" per the abstract)

## Fit to our project

reuse_idea - effort S - gather (cross-attention to features) versus separate (self-attention among siblings) are two roles; our verification and sibling-attention stages play them.

## Needs

a DETR-style decoder.

## Conflicts

none for the idea; code reuse needs the DETR stack.

## Tried in our experiments

none
