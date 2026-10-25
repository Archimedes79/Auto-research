# MH06 - MCL and stochastic MCL (sMCL): oracle loss, winner-take-gradient

*Pass: loop 16 (2026-10-23), cluster `multihyp` (multiple hypotheses, set prediction, matching-free losses). Reading depth: **M (sMCL), A (MCL 2012; PDF not decodable)** (A abstract, M method, C code page). Verified: checked (sMCL), partly (MCL 2012). Pipeline stage: L.*

## Papers

Multiple Choice Learning: Learning to Produce Multiple Structured Outputs (Guzman-Rivera, Batra, Kohli, NIPS 2012) https://proceedings.neurips.cc/paper/2012/hash/cfbce4c1d7c425baf21d6b6f2babe6be-Abstract.html; Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles (Lee, Purushwalkam, Cogswell, Ranjan, Crandall, Batra, NIPS 2016) https://arxiv.org/abs/1606.07839 (read via ar5iv)

## Idea

Train M predictors (or heads) so that per example only the best one is penalised: L_O = sum_i min_m l(y_i, f_m(x_i)). MCL (2012) alternates: assign each example to its best predictor, retrain every predictor to convergence on its examples (structured k-means / hard EM, max-margin bound). sMCL does the assignment inside SGD: per example the gradient goes only to the lowest-loss member, re-assigned every batch.

## Formulation

dL_O/df_m(x_i) = p_im * dl/df_m with p_im = 1 iff m is the arg min (ties arbitrary); a batch passes through all M members, per-example loss per member, backprop only to the minimum member. No diversity term; diversity comes from specialisation. MCL baseline used 5 meta-iterations; sMCL about 5x faster. - input / output: M ensemble members or heads; M hypotheses, no scores.

## Training

regime `wta`; labels = y only. CIFAR-10 (CIFAR10-Quick CNN, batch 350, 5000 iterations), VOC2011 FCN fine-tuning 15 epochs at 1e-5, COCO captioning (neuraltalk2).

## Differentiable?

partial - argmin constant; winner-only gradient.

## Sensors and data

CIFAR-10, Pascal VOC 2011, MSCOCO captions.

## Reported results

oracle CIDEr-D at M = 5 (Table 1): sMCL 0.922, MCL 0.852, Dey et al. 0.910, independent 0.831; VOC2011 about 7 % relative oracle mIoU gain over a classical ensemble; beam search of one model: 0.943 oracle CIDEr-D at width 5 but far fewer unique n-grams. The paper does not discuss member collapse; DiverseNet (MH03) names it.

## Compute

M forward passes; sMCL about 5x faster than MCL.

## Code

not checked

## Fit to our project

reuse_idea - effort S - WTA over b children is `min` over b distances; baseline for all variants.

## Needs

a loss comparable across members; no scores in the original (add MH01).

## Conflicts

with a single GT only one child learns per tip; plain min assumes an oracle selects at test time (our beam needs scores).

## Tried in our experiments

none
