# E45: TopoLogic-style link scoring: learnable distance kernel x semantic similarity of the foundation features at the link ends

Status: completed

**Question:** The learned link head of E42 only tied the rule-based linking. TopoLogic (NeurIPS 2024) combines a geometric part (the distance between the end of one lane and the start of the next through a learnable exponential mapping) with a semantic part (inner product of two learned embeddings) instead of a plain MLP, and SAM-Road decides edges from foundation features sampled at both vertices. Does adding learned semantic embeddings of the frozen foundation features at the two link ends - and TopoLogic's learnable distance kernel - improve the link decisions on the refined pieces?

## Setup
- Candidates, labels and training pieces are those of E42: candidate links (end of piece i -> start of piece j) = inherited grid edges, cut gaps and geometrically plausible pairs; label = directed ground-truth reachability; 112,000 training candidates from the out-of-fold refined pieces of the three cross-fitted dense heads (34,000 + 39,700 + 38,500).
- Features: the 32 engineered link features of E42 (geometry, flags, existence at the ends, observations along the link) PLUS the 39 frozen foundation channels averaged over the last 3 points of piece i and the first 3 points of piece j (78 numbers).
- Model (link_head2.py): geometry branch = MLP (3 hidden layers) on the 32 features; semantic branch = two MLPs (39 -> 64 -> 32) and the scaled inner product of the two 32-d embeddings (TopoLogic's similarity topology; SAM-Road's source / target features); distance term = -(d / sigma)^p with learnable sigma and p (TopoLogic's learnable exponential mapping; start sigma 3 m, p 2); logit = geometry + w_s * similarity + w_d * distance term with learnable w_s, w_d. BCE, false links weighted x2 (errors cost more than omissions), 4,000 steps; the link threshold tau_l is chosen on VAL.
- Evaluation on the refined pieces of the single-frame system of E39 (VAL / TEST), compared with the rule links and the E42 head.

![architecture](architecture.png)

## Numbers

**Link decisions on the refined pieces of the single-frame system (E39); thresholds on VAL**

| link decision | set | F1@1m | route rec | route prec | route F | split F1 3 / 6 m | merge F1 3 / 6 m | dir acc |
|---|---|---|---|---|---|---|---|---|
| rule links (inherited grid edges + distance / heading) | VAL | 0.892 | 0.769 | 0.824 | 0.796 | 0.198 / 0.297 | 0.235 / 0.401 | 0.963 |
| E42 MLP link head, tau_l 0.05 | VAL | 0.892 | 0.771 | 0.830 | 0.799 | 0.195 / 0.293 | 0.235 / 0.406 | 0.963 |
| E42 MLP link head, tau_l 0.3 | VAL | 0.892 | 0.755 | 0.841 | 0.796 | 0.191 / 0.287 | 0.236 / 0.389 | 0.963 |
| E42 MLP link head, tau_l 0.5 | VAL | 0.892 | 0.736 | 0.845 | 0.787 | 0.163 / 0.256 | 0.198 / 0.315 | 0.963 |
| E45 geometry x semantic link head, tau_l 0.05 | VAL | 0.892 | 0.769 | 0.832 | 0.799 | 0.196 / 0.292 | 0.230 / 0.399 | 0.963 |
| E45 geometry x semantic link head, tau_l 0.1 | VAL | 0.892 | 0.765 | 0.834 | 0.798 | 0.192 / 0.289 | 0.230 / 0.396 | 0.963 |
| E45 geometry x semantic link head, tau_l 0.3 | VAL | 0.892 | 0.753 | 0.841 | 0.794 | 0.186 / 0.283 | 0.223 / 0.374 | 0.963 |
| E45 geometry x semantic link head, tau_l 0.5 | VAL | 0.892 | 0.736 | 0.845 | 0.786 | 0.161 / 0.255 | 0.190 / 0.312 | 0.963 |
| rule links (inherited grid edges + distance / heading) | TEST | 0.869 | 0.703 | 0.774 | 0.736 | 0.205 / 0.329 | 0.212 / 0.357 | 0.950 |
| E42 MLP link head, tau_l 0.05 | TEST | 0.869 | 0.703 | 0.778 | 0.738 | 0.193 / 0.317 | 0.207 / 0.355 | 0.950 |
| E42 MLP link head, tau_l 0.3 | TEST | 0.869 | 0.685 | 0.790 | 0.734 | 0.196 / 0.313 | 0.207 / 0.341 | 0.950 |
| E42 MLP link head, tau_l 0.5 | TEST | 0.869 | 0.663 | 0.796 | 0.723 | 0.180 / 0.288 | 0.173 / 0.285 | 0.950 |
| E45 geometry x semantic link head, tau_l 0.05 | TEST | 0.869 | 0.701 | 0.779 | 0.738 | 0.196 / 0.322 | 0.205 / 0.352 | 0.950 |
| E45 geometry x semantic link head, tau_l 0.1 | TEST | 0.869 | 0.696 | 0.783 | 0.737 | 0.195 / 0.320 | 0.202 / 0.347 | 0.950 |
| E45 geometry x semantic link head, tau_l 0.3 | TEST | 0.869 | 0.678 | 0.791 | 0.730 | 0.187 / 0.303 | 0.200 / 0.322 | 0.950 |
| E45 geometry x semantic link head, tau_l 0.5 | TEST | 0.869 | 0.659 | 0.796 | 0.721 | 0.165 / 0.267 | 0.176 / 0.283 | 0.950 |

## Result images

![Link head operating point on VAL: the TopoLogic-style head follows the E42 head and the rule.](e45_links.png)
*Link head operating point on VAL: the TopoLogic-style head follows the E42 head and the rule.*

## Findings
- No gain again. VAL, best thresholds (tau_l 0.05): route F 0.799 against 0.796 for the rule links and 0.799 for the E42 head; split / merge F1 at 6 m 0.292 / 0.399 against 0.297 / 0.401 (rule) and 0.300 / 0.409 (E42). TEST: route F 0.738 against 0.736 (rule), split / merge 0.322 / 0.352 against 0.329 / 0.357. All differences are at most 0.01 and of either sign.
- The semantic branch fits the training candidates a little better than the MLP of E42 (training precision 0.919 / recall 0.851 at tau 0.5 against 0.896 / 0.826) but this does not transfer to the validation scenes: the same seen-to-unseen shift as everywhere in this project (E30) - features of the training scenes are more informative than those of unseen scenes - lets additional capacity memorise rather than generalise.
- Higher thresholds lower recall faster than they raise precision (VAL route F 0.799 -> 0.794 -> 0.786 -> 0.780 for tau_l 0.05 / 0.3 / 0.5 / 0.6, route precision 0.832 -> 0.847): for the route metric a permissive threshold is the better operating point, as in E42.
- Where the gain is NOT: TopoLogic's large gain comes from replacing an MLP over lane queries that ignores geometry by an explicit geometric kernel; our link decision already has the geometry as input. The remaining errors are the annotation conventions (where a lane record ends) and missing pieces. The node-guided assembly of E44 improves the split / merge F1 at 6 m by +0.27 / +0.27 (VAL) with the same pieces, i.e. the information that matters is the learned NODE position, not the choice among candidate links.
- Recommendation: keep the rule links (simplest, equal quality); use the learned key points of E44 for the nodes. The learned semantic similarity is a documented negative result.

## Limits
- The candidate set is the rule's own (inherited + geometric pairs within 8 m); a link that is not a candidate cannot be found. Key-point features (E44) were not used in the head because the key-point head was not cross-fitted (its predictions on the training scenes are optimistic).
- Single seed; trained on out-of-fold pieces of foundation-seen scenes.
- TopoLogic and SAM-Road are trained end to end with their detectors; here only the decision stage is learned on frozen features.
