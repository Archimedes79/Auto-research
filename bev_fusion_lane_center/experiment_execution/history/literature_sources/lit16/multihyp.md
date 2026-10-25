# Cluster MH - multi-hypothesis generation, selection and training that stays trainable (literature, loop 16)

Date: 2026-10-23. Card prefix MH. Status: version 3 (all 58 cards written; sections 3 - 6 in the same version).

Card index: children training / WTA family MH01 - MH09 | multimodal heads in trajectory prediction and planning MH10 - MH18 | assignment variants and Hungarian-free detectors MH19 - MH30 | duplicate removal / rescoring MH31 - MH33 | differentiable selection MH34 - MH48 | beam-aware training, exposure bias, free roll-out MH49 - MH54 | hypothesis trees and tracking MH55 - MH58.

## 1. Coverage statement

- Method: WebSearch and WebFetch only; no downloads, git, curl or pip. Every fetched page comes back as a model-written summary of the page text, so equations and numbers in the cards are transcribed from these summaries; where a summary flagged a gap or corrected my question, the card says "not stated" or reports the correction. `verified: checked` means the page was fetched in this session. Venues and years in the paper lines are from the page when it showed them, otherwise from memory.
- Depth M (method section read via arXiv HTML / ar5iv / a text copy of the paper): rMCL, aMCL (+ README), DiverseNet, relaxed WTA, EWTA, sMCL, Confident MCL, DPP diversity, multi-loss diversity, MultiPath, MultiPath++, MTR, Wayformer, QCNeXt report + Scene Transformer, DenseTNT, DiffusionDrive, PGP, Villaflor et al., POTO / DeFCN, OneNet, YOLOv10 dual assignment, YOLOX SimOTA (partly), Group-DETR, H-DETR, Co-DETR, DAC-DETR, DN-DETR, MapTRv2 one-to-many branch, Align-DETR, Relation Networks, GossipNet / Tnet, SOFT top-k, entmax / sparsemax, OT sort / rank, perturbed optimizers, subset relaxations (Xie-Ermon, SIMPLE), Stochastic Beams, unordered-set estimator, soft beam (Goyal), differentiable beam decoder (Collobert), particle filters (Jonschkowski, Karkus, Corenflos, Scibior-Wood), straight-through family (Gumbel-softmax, Yin, ReinMax), Switch routing, BSO, Negrinho, scheduled sampling, Polygon-RNN++, Yadav (closed-loop multimodal), Diverse Beam Search, Trexplorer Super, Selvan MHT, DTPP.
- Depth A only: original MCL (Guzman-Rivera 2012; the PDF text could not be decoded, its algorithm is taken from the description in the sMCL paper), QCNet paper itself (CVF PDF returned 403; details from the QCNeXt report and the repository README), Ding et al. GCPR 2022 (PDF unreadable, no arXiv version), Kim et al. MHT (ICCV 2015: 403; ECCV 2018 neural gating: abstract only), RefTr, DETA, O2SFormer, DynamicViT (ar5iv failed), Bengio 2013 straight-through, learned beam search on graph RNNs (El Khbir 2024), geometry-aware WTA theory.
- Searched and not found: a differentiable multi-hypothesis tracker (MHT with trainable hypothesis tree); a fixed-fan-out WTA tree for lane graphs; any paper that trains geometric tree nodes through a beam; a withdrawn paper (arXiv 2603.24155) was found and is not used.
- Code URLs are given only where a page or search result showed them (aMCL, QCNet, Confident MCL, Group DETR, DeFCN, DAC-DETR announced, MapTR, O2SFormer, DynamicViT, Trexplorer Super, DiffusionDrive, CL-Traj); READMEs were opened for aMCL and QCNet only; licences are left to cluster EV (the aMCL README states none, the QCNet README states Apache 2.0). Author lists marked "from memory" were not on the page read.

## 2. Cards

### CARD MH01 - Resilient Multiple Choice Learning (rMCL): multi-target WTA + learned score head
- papers: Resilient Multiple Choice Learning: A learned scoring scheme with application to audio scene analysis (Letzelter, Fontaine, Chen, Perez, Essid, Richard, NeurIPS 2023) https://arxiv.org/abs/2311.01052 (full text read via https://ar5iv.labs.arxiv.org/html/2311.01052); theory only (A): Winner-takes-all learners are geometry-aware conditional density estimators (Letzelter et al., ICML 2024) https://arxiv.org/abs/2406.04706
- depth: M (rMCL), A (ICML 2024 theory)
- stage: L, S4
- idea: K hypotheses are trained with a multi-target winner-takes-all loss (each ground-truth target pulls only its nearest hypothesis, i.e. the Voronoi cell it falls in). A separate score per hypothesis is trained with binary cross-entropy to say "this hypothesis wins at least one target of this input". At inference the normalised scores are the probability mass of each cell, so the output is a small weighted set of points; hypotheses sitting in rare regions get low scores instead of looking confident.
- formulation: hypotheses f^k(x), scores gamma^k(x) in [0,1] from separate last layers on a shared representation. Cell Y^k(x) = {y : l(f^k(x), y) < l(f^r(x), y) for all r != k}. L_WTA = sum over targets y in Y_s of sum_k 1[y in Y^k(x_s)] * l(f^k(x_s), y) (targets assigned independently; two targets may share one hypothesis). K+ = hypotheses winning >= 1 target, K- = rest. L_scoring = -( sum_{k in K+} log gamma^k + sum_{k in K-} log(1 - gamma^k) ). Total = L_WTA + beta * L_scoring, beta = 1. rMCL* draws ONE loser uniformly from K- per step (memory saving). Inference: gamma^k <- gamma^k / sum_r gamma^r; p(y|x) = sum_k gamma^k delta(y - f^k(x)) (or uniform inside the cell). The relaxed weights (1 - eps) / eps/(K-1) are NOT part of rMCL (eps = 0.5 only for the eps-WTA baseline and the eps-rMCL combination).
- input: shared trunk + K regression heads + K score heads (toy: 3-layer MLP, 256 hidden, 20 hypotheses; audio: SELDnet-style CRNN with 3 / 5 / 10 / 20 heads).
- output: K points + K scores (sum 1 after normalisation).
- training: loss above; labels = the GT set only (no matching, no order); regime `wta`. Toy: Adam, 100 k samples, batch 1024, 20 epochs. Audio: AdamW, batch 128, lr 0.05 with the Vaswani schedule, epochs not stated.
- differentiable: partial - the argmin / cell assignment is a constant (no gradient through the choice); regression gradient reaches the winner only; whether score gradients are detached from the trunk is not stated.
- sensors_data: synthetic toy data; sound-source localisation (ANSYN anechoic, RESYN reverberant); no vision.
- results: ANSYN, 5 hypotheses, EMD columns D1 / D2 / D3 (as in the paper table): rMCL 7.04 / 13.87 / 20.76; WTA 48.22 / 44.41 / 41.83; top-n WTA 5.14 / 18.09 / 25.12; eps-WTA 5.51 / 19.20 / 28.39. Oracle error: WTA 3.56 / 6.53 / 10.44, rMCL 3.85 / 7.16 / 11.29. Number of hypotheses (rMCL, EMD D3): 3 -> 20.96, 5 -> 20.76, 10 -> 21.28, 20 -> 22.55. Collapse: described as a risk (one or few heads win almost always); not observed in their audio runs (winner histogram in the appendix).
- compute: K heads on one trunk; rMCL* avoids the per-loser cost. No latency numbers.
- code: not found (an author repository exists, from memory; URL not seen)
- fit_to_us: reimplement - effort S - the score means "wins a target" (cell mass), not "lies on a lane", so it cannot replace our validity.
- compat_needs: a distance between a hypothesis and each GT element that is comparable across hypotheses (child endpoint to GT continuation point or polyline); a GT set of size 1 or 2 per tip; hypotheses in one metric output space (ego metres).
- compat_conflicts: (1) with one GT continuation per tip (the common case) b - 1 children get no regression gradient and a score target 0 even when they lie on the lane (cell mass versus validity: a separate on-lane validity head is needed); (2) normalising scores to sum 1 removes independent per-node validity (fork = two true children); (3) no remedy inside the method for starved heads (see MH02 - MH05).
- tried_before: none (E31 / E32 used Hungarian matching for a sparse decoder, dropped)
- verified: checked

### CARD MH02 - Annealed Multiple Choice Learning (aMCL): softmin assignment with a cooling temperature
- papers: Annealed Multiple Choice Learning: Overcoming limitations of Winner-takes-all with annealing (Perera, Letzelter, Mariotte, Cortes, Chen, Essid, Richard, NeurIPS 2024) https://arxiv.org/abs/2407.15580 (text via https://arxiv.org/html/2407.15580, last 78 k characters not read); code https://github.com/Victorletzelter/annealed_mcl
- depth: M+C
- stage: L, T
- idea: Replace the hard min of WTA by a Boltzmann softmin over hypotheses and lower its temperature T during training. At high T all hypotheses are pulled to the conditional mean, they split into subgroups at critical temperatures (phase transitions) and end as a WTA solution at T -> 0. This avoids bad local minima of greedy WTA (hypotheses stuck far from the data, unused hypotheses) without the permanent centroid bias of relaxed WTA.
- formulation: q_T(k | x, y) = exp(-l(f_k(x), y) / T) / sum_s exp(-l(f_s(x), y) / T), l = squared Euclidean distance; L_aWTA = sum_k q_T(k | x, y) * l(f_k(x), y) with q DETACHED. T -> 0: one-hot (MCL); T -> inf: uniform, all hypotheses go to E[Y | x]. First critical temperature T_c = 2 * lambda_max(conditional covariance of Y given x). Schedule: T constant within an epoch; exponential T(t) = T0 * rho^t (UCI: T0 = 0.5, rho = 0.95, 1000 epochs), linear (synthetic: T0 = 1.0, 1000 epochs, 49 hypotheses), several T0 tried for speech; no rule for choosing the schedule (stated as open). Scores: same BCE-on-cell-membership scoring loss as rMCL (MH01).
- input: any regression network with n heads (toy MLP; UCI n = 5; DPRNN on WSJ0-mix). Repo: PyTorch (+ Lightning for separation); `source_separation/losses.py` has PIT, MCL, aMCL, EMD; `models.py` has an extra scoring head; a temperature scheduler component is not described in the README.
- output: n hypotheses (+ n scores).
- training: loss above, GT set only; regime `wta` (annealed).
- differentiable: partial - q detached; gradient reaches all hypotheses weighted by q (more than WTA); final solution is WTA.
- sensors_data: synthetic (three Gaussians), UCI regression, WSJ0-mix speech (2 - 5 speakers; WSJ0 needs LDC access).
- results: UCI distortion (aMCL / MCL / relaxed WTA eps = 0.1): Protein 0.77 / 0.80 / 1.67; Energy 0.28 / 1.25 / 0.30; Yacht 1.15 / 3.09 / 1.34; Boston and Concrete: MCL lower (2.14 vs 2.69; 5.13 vs 5.71). Speech SI-SDR, 2 speakers: PIT 16.88, MCL 16.30 +- 0.59, relaxed WTA 16.70, aMCL 16.85 +- 0.13; 3 speakers 10.01 / 10.06 / 9.43 / 10.00. MCL collapsed for some seeds, aMCL had lower seed variance.
- compute: assignment O(m n) instead of O(m^3) for PIT; slow cooling lengthens training; no wall-clock.
- code: https://github.com/Victorletzelter/annealed_mcl
- fit_to_us: reimplement - effort S - the early high-temperature phase keeps both children of a tip alive; the schedule needs tuning.
- compat_needs: smooth loss l (squared distance); detached assignment weights; an epoch-wise schedule hook.
- compat_conflicts: at high T both children converge to the mean of the GT set at that tip (for a fork: the midpoint between two branches) - harmless early, wrong if cooling stops early; no remedy for the single-GT starvation after cooling (the second child splits off only where data show two modes).
- tried_before: none
- verified: checked

### CARD MH03 - DiverseNet: WTA with several ground truths per input + "catch-up" term
- papers: DiverseNet: When One Right Answer is not Enough (Firman, Campbell, Agapito, Brostow; CVPR 2018, arXiv upload 2020) https://arxiv.org/abs/2008.10634 (read via ar5iv)
- depth: M
- stage: L
- idea: Several valid labels per input are matched to hypotheses independently (each label to its best hypothesis, not one-to-one), and a second term always updates the hypothesis that is farthest from every label, so no hypothesis is starved ("one or more members fail to receive any training signal" is the named failure of stochastic MCL).
- formulation: L_div = sum over (x, Y) sum_{y in Y} min_c l(f(c, x), y); L_catchup = (1/|C|) sum over data of max_c min_{y in Y} l(f(c, x), y); L = L_div + beta * L_catchup, beta = 1 everywhere. With one label per input only one term of L_div remains. Hypotheses come from one network with a control input c (one-hot or sampled vector) concatenated into the activations, not from separate heads; |C| = 8 (MNIST-OCC), 6 (ShapeNet).
- input / output: image or volume -> |C| alternative outputs.
- training: regime `wta` + catch-up; labels = the set of valid answers.
- differentiable: partial - argmin / argmax over hypotheses constant; catch-up gives gradient to the worst hypothesis only.
- sensors_data: MNIST occlusion completion, ShapeNet volumes, traffic-flow prediction.
- results: traffic flow, best endpoint error for k = 1..4 outputs: Lee et al. (sMCL) 0.313 / 0.215 / 0.172 / 0.158; L2 + dropout 0.209 / 0.194 / 0.187 / 0.184; DiverseNet 0.203 / 0.174 / 0.158 / 0.146. ShapeNet: about 12.6 M parameters versus 66.8 M for the tree-net baseline. Some sMCL members "failed to learn, giving null predictions" on MNIST-OCC (no count given).
- compute: single network; not stated beyond parameters.
- code: not checked
- fit_to_us: reimplement - effort S - this is the cleanest published form of "b hypotheses, 1 or 2 valid answers, no matching"; the catch-up term addresses the idle second child directly.
- compat_needs: distance between hypothesis and each valid label; beta weight.
- compat_conflicts: the catch-up term drags the worst child to the nearest label, so at normal tips both children land on the same lane point (redundant children; needs a separate redundancy step such as sibling attention, MH31); with a point-to-polyline loss the pull is to "any point of the lane".
- tried_before: none
- verified: checked

### CARD MH04 - Relaxed WTA (epsilon floor on the non-winners)
- papers: Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses (Rupprecht et al., ICCV 2017; authors and venue from memory) https://arxiv.org/abs/1612.00197 (read via ar5iv)
- depth: M
- stage: L
- idea: Replicate the output layer M times and train with a meta-loss that gives the nearest hypothesis weight 1 - eps and every other hypothesis eps / (M - 1); the small floor keeps starved hypotheses moving until their cell contains data.
- formulation: loss = sum_m w_m(y) * L(h_m(x), y), w = 1 - eps for the nearest, eps/(M-1) for the others (sum 1); eps = 0.05 in all experiments; additionally full predictions are dropped at random with 1 % probability (as relayed by the page summary). Necessary condition for a minimum: every hypothesis equals the conditional mean of its own cell (centroidal Voronoi tessellation). No score branch; uncertainty read from the hypothesis spread. Dead hypotheses: if all start far from the targets one cell captures every label and only that hypothesis learns.
- input / output: shared trunk, M copies of the last layer; any output type (pose, frame, class confidences C * M).
- training: SGD; regime `wta`; GT only.
- differentiable: partial - assignment constant; non-winners get a small gradient.
- sensors_data: LSP human pose, future-frame prediction (synthetic intersection, NTU RGB-D), VOC / COCO multi-label, VOC2012 segmentation.
- results: 4-hypothesis FCN8s on VOC2012 70.3 % mIoU vs 69.1 % for MCL, with 134.9 M vs 539.6 M parameters; pose (strict PCP, oracle): 1 / 2 / 5 / 10 hypotheses 59.7 / 60.0 / 61.2 / 62.8. Cost named: faint "ghost" predictions from centroid bias.
- compute: only the last layer grows with M.
- code: not checked
- fit_to_us: reuse_idea - effort S - one scalar; it pulls the second child toward the first (blur), and gives no score.
- compat_needs: nothing beyond a distance.
- compat_conflicts: pull toward the cell mean blurs the fork angle when eps is large; aMCL (MH02) argues the bias is permanent.
- tried_before: none
- verified: checked

### CARD MH05 - Evolving WTA (EWTA: k winners, k shrinking from M to 1) and sampling + fitting of mixtures
- papers: Overcoming Limitations of Mixture Density Networks: A Sampling and Fitting Framework for Multimodal Future Prediction (Makansi, Ilg, Cicek, Brox, CVPR 2019; authors from memory) https://arxiv.org/abs/1906.03631 (read via ar5iv)
- depth: M
- stage: L, T
- idea: Name why mixture density networks fail (mode collapse, numerical instability in higher dimensions) and split the problem: first generate hypotheses with a WTA-style loss, then fit a mixture on top. To stop WTA hypotheses getting stuck between samples (and relaxed WTA leaving many at the centre) the k nearest hypotheses are updated per target, with k = M at the start, decreased to 1.
- formulation: meta-loss sum_k w_k * l(h_k, y), w_k = 1 for the k closest hypotheses, 0 otherwise (selection by Euclidean distance); k decreases from M to 1 (explicit step schedule not stated). Fitting stage: each hypothesis gets an M_c-dim softmax assignment gamma, weights = mean assignment, means = assignment-weighted means, variances by the law of total variance; two FC layers + dropout 0.5; stages trained one after the other, then EWTA removed and everything fine-tuned with NLL. K = 40 hypotheses, 4 mixture components (SDD table: 20 hypotheses).
- input / output: FlowNetS encoder + conv layers -> K position hypotheses; mixture of 4 Gaussians (Laplace on SDD).
- training: EWTA stage 250 k iterations, fitting added at 350 k, joint fine-tuning after 550 k (MDN baseline 150 k + 200 k); regime `wta`.
- differentiable: partial - top-k selection constant; fitting stage fully differentiable.
- sensors_data: synthetic CPI dataset (acronym expansion not on the page read), Stanford Drone Dataset.
- results: CPI Table 1 (oracle error / EMD): WTA 6.96 / 3.94, relaxed WTA 7.94 / 2.82, EWTA 9.84 / 1.89 (lower EMD = hypotheses match the true distribution better although the oracle error is higher). Table 2 NLL / EMD: MDN 9.20 / 1.83, EWTAD-MDF 8.33 / 1.57. SDD NLL: MDN 9.71, EWTAD-MDF 9.33.
- compute: single forward pass at inference; hardware not reported.
- code: not checked
- fit_to_us: reuse_idea - effort S - for b = 2 the schedule k = 2 -> 1 is a two-stage form of annealing.
- compat_needs: ranking of the b hypotheses by distance.
- compat_conflicts: k winners per target put k children onto ONE target early; with b = 2 that is exactly the "both children on the same lane" state, wanted only early.
- tried_before: none
- verified: checked

### CARD MH06 - MCL and stochastic MCL (sMCL): oracle loss, winner-take-gradient
- papers: Multiple Choice Learning: Learning to Produce Multiple Structured Outputs (Guzman-Rivera, Batra, Kohli, NIPS 2012) https://proceedings.neurips.cc/paper/2012/hash/cfbce4c1d7c425baf21d6b6f2babe6be-Abstract.html; Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles (Lee, Purushwalkam, Cogswell, Ranjan, Crandall, Batra, NIPS 2016) https://arxiv.org/abs/1606.07839 (read via ar5iv)
- depth: M (sMCL), A (MCL 2012; PDF not decodable)
- stage: L
- idea: Train M predictors (or heads) so that per example only the best one is penalised: L_O = sum_i min_m l(y_i, f_m(x_i)). MCL (2012) alternates: assign each example to its best predictor, retrain every predictor to convergence on its examples (structured k-means / hard EM, max-margin bound). sMCL does the assignment inside SGD: per example the gradient goes only to the lowest-loss member, re-assigned every batch.
- formulation: dL_O/df_m(x_i) = p_im * dl/df_m with p_im = 1 iff m is the arg min (ties arbitrary); a batch passes through all M members, per-example loss per member, backprop only to the minimum member. No diversity term; diversity comes from specialisation. MCL baseline used 5 meta-iterations; sMCL about 5x faster.
- input / output: M ensemble members or heads; M hypotheses, no scores.
- training: regime `wta`; labels = y only. CIFAR-10 (CIFAR10-Quick CNN, batch 350, 5000 iterations), VOC2011 FCN fine-tuning 15 epochs at 1e-5, COCO captioning (neuraltalk2).
- differentiable: partial - argmin constant; winner-only gradient.
- sensors_data: CIFAR-10, Pascal VOC 2011, MSCOCO captions.
- results: oracle CIDEr-D at M = 5 (Table 1): sMCL 0.922, MCL 0.852, Dey et al. 0.910, independent 0.831; VOC2011 about 7 % relative oracle mIoU gain over a classical ensemble; beam search of one model: 0.943 oracle CIDEr-D at width 5 but far fewer unique n-grams. The paper does not discuss member collapse; DiverseNet (MH03) names it.
- compute: M forward passes; sMCL about 5x faster than MCL.
- code: not checked
- fit_to_us: reuse_idea - effort S - WTA over b children is `min` over b distances; baseline for all variants.
- compat_needs: a loss comparable across members; no scores in the original (add MH01).
- compat_conflicts: with a single GT only one child learns per tip; plain min assumes an oracle selects at test time (our beam needs scores).
- tried_before: none
- verified: checked (sMCL), partly (MCL 2012)

### CARD MH07 - Confident MCL (non-winners pushed to uniform)
- papers: Confident Multiple Choice Learning (Lee, Hwang, Park, Shin, ICML 2017) https://arxiv.org/abs/1706.03475 (read via ar5iv); code https://github.com/chhwang/cmcl (named in a search result; Python 2.7 / TensorFlow per the summary)
- depth: M
- stage: L
- idea: MCL members stay confidently wrong on inputs they do not specialise in, so averaging them fails. The winner keeps the task loss; every other member gets a KL divergence to the uniform distribution (weight beta), which makes losers low-confidence.
- formulation: confident oracle loss: for the member that best fits the example the usual task loss; for the others beta * KL(U || p_m); stochastic labeling replaces the exact KL gradient by a cross-entropy against sampled uniform labels; feature sharing: each member sees a random masked subset (probability lambda = 0.7) of the others' lower-layer features (before the first pooling); beta in {0.5 ... 1.5}, overlap K in {2, 3, 4}.
- input / output: classifier ensemble (or FCN for segmentation); class posteriors.
- training: regime `wta`; classification labels.
- differentiable: partial - winner choice constant; loser term differentiable.
- sensors_data: CIFAR-10, SVHN, iCoseg.
- results: CIFAR-10, 5 ResNet-20: MCL ensemble-average top-1 error 60.40 %, independent ensemble 15.34 %, CMCL 8.75 %; SVHN 3.68 %.
- compute: not stated.
- code: https://github.com/chhwang/cmcl
- fit_to_us: reuse_idea - effort S - the loser-side rule for scores: a child that is not responsible for a GT must be low-confident only when it is really off-lane; for on-lane losers the uniform-KL rule would be wrong (they are valid).
- compat_needs: per-member probabilities.
- compat_conflicts: designed for class posteriors; no analogue for geometry.
- tried_before: none
- verified: checked

### CARD MH08 - Repulsion / diversity terms for a set of hypotheses (DPP loss; pairwise-distance loss)
- papers: Diverse Trajectory Forecasting with Determinantal Point Processes (Yuan, Kitani, ICLR 2020) https://arxiv.org/abs/1907.04967 (ar5iv); A Multi-Loss Strategy for Vehicle Trajectory Prediction: Combining Off-Road, Diversity, and Directional Consistency Losses (2024) https://arxiv.org/abs/2411.19747 (HTML)
- depth: M
- stage: L
- idea: Add an explicit set-level term that keeps K hypotheses apart. DPP: a kernel from pairwise distances, loss = negative expected cardinality of the DPP. Multi-loss paper: sum of pairwise trajectory distances among drivable modes, applied only during a short fine-tuning.
- formulation: DPP: similarity S_ij = exp(-k * d_ij^2) from pairwise trajectory distances gives the DPP kernel L; loss = -tr(I - (L + I)^-1) (negative expected cardinality), preferred to the DPP log-likelihood which becomes unstable for similar samples; a "quality" term is defined in latent space (codes inside a sphere containing 90 % of prior samples); k = 1 (synthetic), 1e-2 (mocap), 5 (Human3.6M); the sampling function is trained with the decoder fixed; greedy DPP MAP at inference. Multi-loss: Mode Diversity = sum over drivable pairs (1/T) sum_t ||y_t^i - y_t^j||_2 (sign convention not stated, presumably maximised); plus off-road hinge (signed distance + margin) and a centreline direction-consistency term; weight values not given.
- input / output: K trajectories or latent codes.
- training: added to WTA / minADE; regime `wta`.
- differentiable: yes (the kernel / distance is differentiable in the hypotheses).
- sensors_data: synthetic 2-D vehicle, motion capture, Human3.6M (DPP); Wayformer and AutoBots on nuScenes and Argoverse 2 (multi-loss).
- results: multi-loss, Wayformer on nuScenes: diversity 57 -> 74 while minADE 1.08 -> 1.10, minFDE 2.47 -> 2.55, MR 0.42 -> 0.46; AutoBots / Argoverse 2: minADE 0.92 -> 0.87, diversity 58 -> 62. The paper says adding the diversity loss from the start "tends to destabilize training", so it is used for 10 fine-tuning epochs. DPP paper: DSF beats cVAE, MCL, cGAN on ADE / FDE and on repetitiveness (ASD / FSD).
- compute: not stated.
- code: not checked
- fit_to_us: reuse_idea - effort S - a hinge on the child-child distance gated to tips with two GT branches (or to a late stage) is the cheap way to keep fork children apart.
- compat_needs: pairwise distance matrix of the b children (b is tiny).
- compat_conflicts: repulsion at normal tips pushes a valid second child off the lane; unstable if active from step 0.
- tried_before: none
- verified: checked

### CARD MH09 - Diverse Beam Search (group-wise beam with a dissimilarity penalty)
- papers: Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models (Vijayakumar et al., 2016) https://arxiv.org/abs/1610.02424 (read via ar5iv)
- depth: M
- stage: S4
- idea: Split the B beams into G groups; each group runs ordinary beam search, but its scores get a penalty for candidates already chosen by earlier groups. Inference-time only; no retraining.
- formulation: score = log-prob(candidate) - lambda * Delta(candidate, tokens chosen by previous groups at this step); Hamming diversity is the best simple form; G = B gives the most exploration; lambda in 0.2 - 0.8 worked; G = 1 is standard beam search. About T + G steps if groups are staggered in parallel.
- input / output: any sequence model; B diverse sequences.
- training: none.
- differentiable: no (selection top-k; inference-time).
- sensors_data: COCO / PASCAL-50S captioning, VQG, WMT en-fr (B = 20).
- results: COCO captioning oracle SPICE@20 28.09 vs 27.34 (standard beam); distinct 4-grams 5.67 -> 13.44; MT oracle BLEU-4@20 18.77 vs 18.44.
- compute: little over standard beam search.
- code: not checked
- fit_to_us: reuse_idea - effort S - a cheap rule to avoid M beam nodes collapsing onto one lane: penalise a candidate by its closeness to already selected nodes (a hand-made NMS inside the beam).
- compat_needs: a dissimilarity function between candidates (distance in metres).
- compat_conflicts: hand rule, no gradient; competes with learned sibling attention (MH31).
- tried_before: none
- verified: checked

### CARD MH10 - MultiPath (fixed anchors: classify the anchor, regress the offset)
- papers: MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction (Chai, Sapp, Bansal, Anguelov, CoRL 2019) https://arxiv.org/abs/1910.05449 (arXiv HTML)
- depth: M
- stage: L, S4
- idea: Fix K anchor trajectories before training; the model outputs a softmax over anchors and, per anchor, a Gaussian offset. Each GT is assigned to the single nearest anchor, so the assignment is a fixed geometric rule and the mixture cannot collapse.
- formulation: k_hat = argmin_k ||GT - a^k||_2 (state-sequence space, canonical agent frame). Loss = -[ log pi(a^{k_hat} | x) + sum_t log N(s_t | a_t^{k_hat} + mu_t, Sigma_t) ] (Eq. 3; regression only for the assigned anchor; with Sigma frozen an L2 loss). Mixture p(s | x) = sum_k pi(a^k | x) prod_t N(s_t | a_t^k + mu_t^k, Sigma_t^k), weights fixed over time. Anchors: k-means on training trajectories (redundant on some data -> uniform sampling of trajectory space). K = 16 (mu, Sigma) or 64 (mu) on the driving data; K = 3 toy; Stanford Drone 64 straight + 1 stationary. Inference: MAP trajectory per anchor weighted by pi; keep the top kappa (no NMS / EM).
- input / output: rasterised scene + state history -> K anchor probabilities + offsets.
- training: 500 k steps, batch 32; regime `wta` with a fixed (anchor) assignment.
- differentiable: yes for all heads; the anchor label is precomputed.
- sensors_data: an internal driving dataset (not public), Stanford Drone; BEV raster.
- results: Table 1 on the driving data: log-likelihood 4.37 (MultiPath mu, Sigma) vs 4.26 (min-of-K) vs 3.64 (regression); minADE5 0.58 (MultiPath mu) vs 0.87 (min-of-K mu). Toy: min-of-K collapsed to 1 or 2 active modes across trials.
- compute: one forward pass; K up to 64.
- code: not checked
- fit_to_us: reuse_idea - effort S - fixed bearing / step-class anchors in the tip frame give every child slot a fixed role; the E94 head is already an anchor-style classifier.
- compat_needs: an anchor set in the tip frame (bearings, step classes); a cross-entropy target "nearest anchor".
- compat_conflicts: needs K >> b (16 - 64 anchors) and then a top-b with suppression; anchors fix the mode count; [own inference] a hard nearest-anchor label flips at sector borders, and a straight lane lies exactly on the border between left and right sectors.
- tried_before: E94 (partial: bearing-bin + step-class classification, teacher forced, one child + fork event); verdict F1@1 0.851 vs 0.867 for rules
- verified: checked

### CARD MH11 - MultiPath++ (learned anchors, hard assignment by distance, aggregation of many modes to few)
- papers: MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction (Varadarajan et al., ICRA 2022) https://arxiv.org/abs/2111.14973 (arXiv HTML; last 3.5 k characters not read)
- depth: M
- stage: L, S4
- idea: Replace fixed anchors by learned, input-independent anchor embeddings (one per mixture component); train with the hard assignment of the GT to the component with the smallest Euclidean distance; at the end reduce a large union of modes (several heads) to the wanted few by a likelihood-weighted greedy selection plus EM-style updates.
- formulation: M embeddings e_1..e_M conditioned on the scene by multi-context-gating blocks; an MLP outputs means, stds, correlation and log-weights per mode (weights tied over time). Loss: maximise the likelihood of the GT under the hard-assigned component (smallest Euclidean distance); Adam, lr 2e-4, batch 64, decay 0.5 every 2 epochs, 800 k steps. Aggregation: bag E heads of L modes -> M' = L * E modes with likelihood / E; greedy centroids (each maximises the likelihood mass within tau of the selected set; submodular, >= 1 - 1/e of optimum), then EM-style updates with hard responsibilities; trained end to end as a final layer; M = 6.
- input / output: encoded agent / road / history contexts -> M trajectory modes with Gaussians and weights.
- training: regime `wta` (hard assignment), no matching between modes and GT.
- differentiable: partial - assignment constant; aggregation is a trainable final layer per the paper.
- sensors_data: Waymo Open Motion, Argoverse.
- results: Waymo validation ablation (k = 6, 8 s): learned anchors minDE 2.305 / MR 0.44 vs static k-means anchors 2.99 / 0.578; test leaderboard (Nov 2021) mAP 0.409, minADE 0.556, MR 0.134. The paper reports that learned anchors now beat static ones and lists richer inputs, architecture and larger batches as possible reasons (study left open).
- compute: not stated.
- code: not checked
- fit_to_us: reuse_idea - effort S - per-slot child queries with hard nearest assignment is the cheapest WTA implementation; aggregation corresponds to our "redundancy suppression".
- compat_needs: per-slot queries conditioned on the tip context; Euclidean distance to the GT.
- compat_conflicts: aggregation is likelihood-based and run at inference; no diversity beyond WTA.
- tried_before: none
- verified: checked

### CARD MH12 - MTR (64 static intention queries from k-means, nearest-to-GT positive, NMS to 6)
- papers: Motion Transformer with Global Intention Localization and Local Movement Refinement (Shi, Jiang, Dai, Schiele, NeurIPS 2022; authors from memory) https://arxiv.org/abs/2209.13508 (arXiv HTML)
- depth: M
- stage: S4, L
- idea: One query per motion mode - K = 64 intention points from k-means on GT endpoints (per category) - plus one dynamic searching query per mode that moves to the endpoint predicted by the previous decoder layer. The mode whose intention point is nearest to the GT endpoint is the positive: only it gets the regression loss, its mixture probability gets cross-entropy. A rule-based NMS picks 6 of 64.
- formulation: static query = MLP(PE(intention point)); dynamic query for layer j+1 = MLP(PE(predicted endpoint of layer j)), gathers features from L = 128 nearest map polylines. Per layer a Gaussian mixture; positive = nearest intention point to the GT endpoint (hard); loss = NLL of the GT under the positive component + cross-entropy on its mixture probability + auxiliary L1 on a dense future head; equal weights. No separate classification head. Inference NMS: top 6 of 64 by endpoint distance, threshold 2.5 m (the authors note it may hurt minADE / minFDE).
- input / output: scene tokens (agent + 768 map polylines, D = 256, 6 + 6 layers) -> 64 trajectories + probabilities.
- training: AdamW, lr 1e-4, batch 80, 30 epochs on 8 GPUs, lr halved every 2 epochs from epoch 20, weight decay 0.01; regime `wta` with a fixed (anchor) assignment.
- differentiable: yes for heads; assignment and NMS constant.
- sensors_data: Waymo Open Motion; vector map and agents (no sensors).
- results: Waymo marginal test mAP 0.4129, minADE 0.6050, minFDE 1.2207, MR 0.1351. Ablation (validation subset): learnable latent queries mAP 0.2633 vs static intention queries 0.3059; uniform grid vs k-means intention points 0.3178 vs 0.3437; full model 0.3437. The paper does not say "mode collapse"; it says direct regression converges slowly and predicts the most frequent modes.
- compute: about 59 ms (batch 1) / 33 ms per scene (batch 6) on an RTX 8000; parameters not given.
- code: not checked
- fit_to_us: reuse_idea - effort S - "intention points" = a fixed candidate set in the tip frame (our teardrop ring samples are one); nearest-to-GT positive + classification over candidates trains the responsible child matching-free.
- compat_needs: a candidate set with fixed geometry in the tip frame; many more candidates than children and a final suppression.
- compat_conflicts: NMS is a non-differentiable inference step; 64 -> 6 contradicts a tiny fixed b unless the pool is the verified ray endpoints and b is taken after suppression.
- tried_before: E94 (partial, see MH10)
- verified: checked

### CARD MH13 - Wayformer (64 learned queries, winner-take-all NLL, aggregation to 6)
- papers: Wayformer: Motion Forecasting via Simple & Efficient Attention Networks (Nayakanti et al., 2022) https://arxiv.org/abs/2207.05844 (arXiv HTML; venue not checked)
- depth: M
- stage: S4, L
- idea: A transformer decoder with k learned initial queries (one per mixture component) cross-attending to the scene encoding; the loss splits into a classification term for the winner and a regression term for the winner only. Benchmarks use 64 components reduced to 6 by the MultiPath++ aggregation.
- formulation: winner = Gaussian whose mean is closest to the GT (distance not named); loss = -log Pr(winner | scene) - log Pr(GT | winner Gaussian) (hard WTA). Aggregation (appendix): greedy centroids that cover the most uncovered likelihood within distance D (2.3 Waymo, 2.9 Argoverse), then k-means style refinement (3 or 5 iterations), at most 6 trajectories; ensembles merge N = 3 decoders (Waymo) or 10 decoders x 15 replicas (Argoverse).
- input / output: scene tokens (early / late / hierarchical fusion, latent-query bottleneck) -> 6 or 64 modes.
- training: 1 M steps, AdamW, lr 2e-4 linear decay, batch 256 (ablations; hyper-parameter ranges in text and appendix are inconsistent).
- differentiable: yes for heads; winner constant; aggregation outside the gradient path as read.
- sensors_data: Waymo Open Motion, Argoverse; no raw sensors.
- results: WOMD (early fusion; latent query vs factorised) minADE 0.545 / 0.545, minFDE 1.128 / 1.126, MR 0.123, mAP 0.419 / 0.412 (MultiPath++ 0.409); Argoverse Brier-minFDE 1.7408 (MultiPath++ 1.7932). No "6 modes directly vs 64 + aggregation" comparison and no collapse numbers on the page.
- compute: ablation models 0.3 - 20 M parameters; latent queries 2x - 16x faster.
- code: not checked
- fit_to_us: reuse_idea - effort S - confirms "train with many more hypotheses than needed, reduce by likelihood-aware aggregation".
- compat_needs: mode-wise distance and probability.
- compat_conflicts: needs a large mode pool; aggregation is inference-time clustering.
- tried_before: none
- verified: checked

### CARD MH14 - QCNet / QCNeXt and Scene Transformer (scene-level WTA, mixture of Laplace, detached proposals)
- papers: Query-Centric Trajectory Prediction (Zhou et al., CVPR 2023) https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Query-Centric_Trajectory_Prediction_CVPR_2023_paper.html (A only); QCNeXt: A Next-Generation Framework For Joint Multi-Agent Trajectory Prediction (technical report) https://arxiv.org/abs/2306.10508 (read; describes QCNet briefly); Scene Transformer: A unified architecture for predicting multiple agent trajectories (Ngiam et al.) https://arxiv.org/abs/2106.08417 (read)
- depth: M+C (QCNeXt report, Scene Transformer, QCNet README); A for the QCNet paper itself
- stage: L, S4, S5
- idea: Proposal module with anchor-free mode queries repeated recurrently, then an anchor-based refinement that takes the (detached) proposals as anchors; mixture of Laplace; winner = best mode at scene level (min displacement). Scene Transformer: F = 6 joint futures, each gets an ID embedding, a softmax over futures, loss = min over futures of the summed per-agent loss.
- formulation: L = L_propose + L_refine + L_cls; L_cls = NLL of the mixing coefficients; winner by scene-level WTA (QCNeXt); proposals have gradients detached before refinement; 3 recurrent steps of 2 s each. Scene Transformer: Laplace-KL loss on position (Huber only on heading), minimum over futures (joint) or per agent (marginal), classifier on the best future weighted 0.1 vs regression 1.0; gradients flow only through the winning future; detaching of the winner choice not stated in either paper.
- input / output: query-centric scene tokens -> K modes with Laplace parameters and mixing weights.
- training: QCNeXt AdamW, 50 epochs, batch 32, cosine lr 5e-4 -> 0, dropout / weight decay 0.1, hidden 128; regime `wta`.
- differentiable: partial - winner constant; proposal -> refinement gradient cut by design.
- sensors_data: Argoverse 1 / 2, Waymo; vectors only.
- results: Argoverse 2 validation single-agent: QCNet minFDE6 1.28, MR6 0.16; QCNeXt 1.24 / 0.15; multi-agent leaderboard QCNeXt avgMinFDE6 1.13 (1.02 with 8-seed ensemble). Scene Transformer Argoverse test (marginal) minADE 0.80, minFDE 1.23, MR 0.13; WOMD marginal vehicle minADE 1.17, minFDE 2.48, MR 0.19. Neither page discusses mode collapse.
- compute: QCNet README: example training on 8 x RTX 3090 with about 160 GB GPU memory in total (batch 4 per device); latency not reported.
- code: https://github.com/ZikangZhou/QCNet (README: PyTorch + PyG + Lightning, Apache 2.0 stated; the README does not describe the loss or the number of modes)
- fit_to_us: reuse_idea - effort S - the "propose then refine, detach the proposal as the anchor of the next stage" pattern is the training form of our generate -> refine cascade.
- compat_needs: refinement that takes proposals as anchors; scene-level winner.
- compat_conflicts: scene-level (joint) WTA couples all agents; for a tree it would couple all tips (not wanted: each tip is independent).
- tried_before: E39 (cascade refiner, +0.001 F1)
- verified: partly (QCNet itself A)

### CARD MH15 - DenseTNT (dense goal candidates, heat map, learned goal-set selection instead of NMS)
- papers: DenseTNT: End-to-end Trajectory Prediction from Dense Goal Sets (Gu, Sun, Zhao, ICCV 2021) https://arxiv.org/abs/2108.09640 (read via ar5iv; venue from memory); TNT: Target-driveN Trajectory Prediction (Zhao et al.) https://arxiv.org/abs/2008.08294 (A)
- depth: M (DenseTNT), A (TNT)
- stage: S4, L
- idea: Score many lane-aligned goal candidates with a softmax heat map (CE against the candidate nearest the GT endpoint), then replace NMS by a learned goal-set predictor that outputs K goals; because only one future is observed, set labels come from an offline optimiser that selects the K goals with minimal expected error under the heat map.
- formulation: candidates within 3 m of centrelines within 50 m of the target, 1 m spacing; goal probability = softmax over candidates, BCE against a one-hot target at the candidate nearest the GT endpoint (Eqs. 3 - 4). Goal-set predictor: 12 heads, each with a one-layer self-attention encoder + max pool and a 2-layer MLP decoder, each outputting K goals + a confidence; losses: BCE on head confidences (best head = 1) + L1 to the pseudo-label. Pseudo-labels: offline hill-climbing minimising E[d(y_hat, Y)] under the heat map (perturbations L = 100 per step, accepts worse moves with probability 0.01). NMS fails because the threshold depends on the heat map and it is greedy. K not specified in the text read.
- input / output: vector scene tokens -> dense goal probabilities -> K goals -> trajectories.
- training: batch 64, hidden 128, 16 epochs stage 1 + 6 epochs stage 2, lr 1e-3, no augmentation; regime `wta`-like (nearest candidate) + pseudo-label regression.
- differentiable: partial - the pseudo-label optimiser is offline; the learned predictor is differentiable.
- sensors_data: Argoverse (2 s history, 3 s future), Waymo 2021 challenge.
- results: Argoverse validation: offline optimiser minADE 0.80, minFDE 1.27, MR 7.0 %; online goal-set predictor 0.82 / 1.37 / 7.0 %; Waymo 2021 challenge 1st place mAP 0.3281. Ablation (Table 2): dense heat maps beat sparse; variety loss gives MR 13.3 %. Offline optimisation about 100 ms per sample on 8 CPUs.
- compute: online latency not reported.
- code: not checked
- fit_to_us: reuse_idea - effort M - "dense candidate heat map + learned selection of a small set trained on pseudo-labels from an offline optimiser" is a learned replacement for our NMS with a matching-free label; our ray-ring samples are the dense candidates.
- compat_needs: dense candidates with a softmax score; an offline set optimiser for labels.
- compat_conflicts: the pseudo-label step is not end to end; 2-stage training.
- tried_before: none
- verified: checked

### CARD MH16 - DiffusionDrive (k-means anchors, nearest-anchor positive, BCE confidences, truncated diffusion)
- papers: DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving (Liao et al., CVPR 2025; venue not on the page read) https://arxiv.org/abs/2411.15139 (arXiv HTML)
- depth: M
- stage: S4, S5, L
- idea: Start the diffusion from noised k-means anchors instead of pure noise (truncated schedule), denoise in 2 steps with a cascade decoder that samples BEV features at the trajectory coordinates, and output a confidence per anchor trajectory; the highest-confidence trajectory is executed.
- formulation: anchors N = 20 (NAVSIM), 18 (nuScenes); noised anchors tau_k = sqrt(abar_i) a_k + sqrt(1 - abar_i) eps, truncated at 50/1000. Label: the anchor closest to the GT trajectory is positive (y_k = 1), all others negative. L = sum_k [ y_k * L1(tau_hat_k, tau_gt) + lambda * BCE(s_hat_k, y_k) ]: negatives get only the classification term. Decoder: deformable spatial cross-attention to BEV features at the trajectory coordinates, cross-attention to agent / map queries, FFN, timestep modulation; two cascade layers with shared parameters; DDIM update.
- input / output: BEV features from the backbone (NAVSIM ResNet-34, nuScenes ResNet-50; the sensor set is not stated on the page read) -> anchor trajectories + confidences.
- training: 100 epochs navtrain, AdamW lr 6e-4, batch 512, 8 x RTX 4090; regime `wta` with fixed anchors + `denoising`.
- differentiable: yes (assignment is a precomputed label).
- sensors_data: NAVSIM (ResNet-34), nuScenes open-loop (ResNet-50).
- results: NAVSIM navtest PDMS 88.1 (Transfuser baseline +4.1); denoising steps 1 / 2 / 3: 87.9 / 88.1 / 88.1; cascade layers 1 / 2 / 4: 87.4 / 88.1 / 88.2; sampled noises 10 / 20 / 40: 84.9 / 88.1 / 88.2; nuScenes open-loop L2 0.57 m, collision 0.08 % (VAD 0.72 m, 0.22 %). The page has no ablation of the number of anchors.
- compute: about 60 M parameters; 45 FPS (NAVSIM, 4090), planning module 7.6 ms; 8.2 FPS on nuScenes.
- code: https://github.com/hustvl/DiffusionDrive
- fit_to_us: reuse_idea - effort M - the closest published match of "k-means anchors in metres + nearest-anchor positive + per-anchor BCE confidence + BEV sampling at the hypothesis coordinates" with a nuScenes BEV backbone; a truncated-diffusion refinement is an alternative to E120.
- compat_needs: anchor set; BEV features sampled at coordinates (we have bilinear sampling).
- compat_conflicts: the published model trains its BEV backbone end to end (ours is frozen and cached); one positive per sample only (a fork needs two positives).
- tried_before: none
- verified: checked

### CARD MH17 - PGP (policy over lane-graph edges, behaviour-cloning label, sampled traversals, K-means)
- papers: Multimodal Trajectory Prediction Conditioned on Lane-Graph Traversals (Deo, Wolff, Beijbom, CoRL 2021) https://arxiv.org/abs/2106.15004 (read via ar5iv)
- depth: M
- stage: S4, S6, L
- idea: A discrete policy over the successor edges of a lane graph is trained by behaviour cloning on the edges the future visits, sampled for many rollouts at inference, and each rollout conditions a trajectory decoder; the final modes are K-means clusters of many samples.
- formulation: nodes = centreline snippets <= 20 m, 1 m poses; policy header = MLP scoring each outgoing edge from motion encoding and node encodings, softmax over outgoing edges plus an edge to an end state; label: each future pose assigned to the closest lane node whose direction is within a yaw threshold; edge visited if both endpoints visited; loss = sum of -log pi over visited GT edges (Eq. 3). Decoder: latent z ~ N(0, I), 5-d; regression loss = min over modes of average displacement error (WTA). 200 samples clustered to K = 10 (results at K = 5 and 10). Pre-train 100 epochs on GT traversals, fine-tune 100 epochs on policy-sampled paths (about 1 min vs 5 min per epoch).
- input / output: HD-map lane graph + agent histories -> traversals -> trajectories.
- training: Adam 1e-4, batch 32, 4 V100; regime `teacher_forcing` (pre-train) then `free_rollout`-like fine-tuning on sampled paths.
- differentiable: partial - sampling of traversals is discrete; WTA regression on the sampled modes.
- sensors_data: nuScenes (HD map input).
- results: nuScenes Table 1: MinADE5 1.30, MinADE10 1.00, MissRate5,2 0.61, MissRate10,2 0.37, off-road rate 0.03. Traversal + latent decoder reaches 1.41 distinct final lanes vs 1.22 latent-only. The paper notes sampling is often redundant; no collapse analysis, no greedy-vs-sampled comparison.
- compute: many samples run in parallel; no latency numbers.
- code: not checked
- fit_to_us: reuse_idea - effort M - a lane-graph fork is trained by behaviour cloning on edges; shows sampling + clustering as the standard way to get K diverse hypotheses from a branching policy; the sample-then-cluster step corresponds to our DAG merge.
- compat_needs: a lane graph as input (we produce it); node snippets.
- compat_conflicts: input is the HD map (ours is the output); sampling is not differentiable.
- tried_before: none
- verified: checked

### CARD MH18 - Learned anchor embeddings as discrete behaviour modes held fixed along an autoregressive rollout
- papers: Tractable Joint Prediction and Planning over Discrete Behavior Modes for Urban Driving (Villaflor et al., ICRA 2024) https://arxiv.org/abs/2403.07232 (arXiv HTML)
- depth: M
- stage: S4, T
- idea: K = 8 learned anchor embeddings are the discrete behaviour modes of each agent; training is open-loop WTA with cross-entropy on the mode logits; at planning time the one-step decoder is rolled out autoregressively with the mode held fixed, so no retraining is needed.
- formulation: z in {1..K}, anchor embeddings in R^{K x D} added to agent features (one set for ego, one for others); each mode predicts a trajectory with Gaussian waypoints (NLL), WTA: only the closest of the K predictions backpropagates the regression loss (per agent); cross-entropy on mode logits with a one-hot at the closest prediction; the decoder predicts H steps open-loop as an auxiliary task. Rollout: first predicted step only, mean output, z_0 = ... = z_{T-1}; planner evaluates only the K ego modes with N = 8 rollouts each (K^T joint sequences judged too expensive).
- input / output: encoded agents and map -> K mode trajectories.
- training: AdamW lr 2e-4 cosine, 100 epochs; regime `wta` (open loop), used `free_rollout` only at test time.
- differentiable: partial - WTA constant.
- sensors_data: CARLA (merging, Longest6); ~1.9 M and ~7.4 M parameters.
- results: merging 96.5 % success vs 85.5 % autopilot; Longest6 driving score 51.0 vs autopilot 44.6, PlanT 37.1. Mode collapse not discussed; discrete modes give more diverse proposals than a continuous latent CVAE.
- compute: planning linear in agents and horizon; no runtime numbers.
- code: not checked
- fit_to_us: reuse_idea - effort S - shows that WTA-trained mode slots can be unrolled for many steps with the slot held fixed (cheap consistency), relevant to a child carrying its slot identity down the tree.
- compat_needs: mode embedding conditioning; a one-step decoder.
- compat_conflicts: holding the mode fixed is a modelling choice (works "empirically"); no tree branching or pruning.
- tried_before: none
- verified: checked

### CARD MH19 - DeFCN / POTO (prediction-aware one-to-one assignment without Hungarian, 3D max filtering)
- papers: End-to-End Object Detection with Fully Convolutional Network (Wang et al., CVPR 2021) https://arxiv.org/abs/2012.03544 (read via ar5iv)
- depth: M
- stage: L, S4
- idea: In a fully convolutional detector, replace the many positives per object by exactly one positive chosen by a prediction-aware quality (class score and IoU, within a spatial prior); add a 3D max filter over neighbouring locations and scales so duplicates are suppressed inside the network; use an auxiliary one-to-many loss for dense supervision. NMS is not needed.
- formulation: Q = 1[pred in Omega_i] * p^(1 - alpha) * IoU^alpha, alpha = 0.8; Omega_i = FCOS-style centre sampling; multiplicative fusion beats additive by 0.7 mAP; alpha = 0 or 1 close the NMS gap less. Selection is an argmax over assignments of the summed Q (the page could not confirm per-GT argmax vs joint). 3D max filter: resample features from tau = 2 neighbouring scales to the current size, then max over scales and a 3 x 3 neighbourhood per channel, GroupNorm + sigmoid. Auxiliary loss: top-9 candidates per FPN level by Q, positives above mean + std of Q, focal loss; training only.
- input / output: FPN feature maps -> box + score maps.
- training: FCOS / Detectron2 2x recipe; regime `matching_free` (quality-based one-to-one) + auxiliary one-to-many.
- differentiable: yes (assignment is a constant label; the max filter passes gradient to the max).
- sensors_data: COCO, CrowdHuman (images).
- results: COCO val R50 without NMS: POTO 38.0; + 3DMF 39.8; + aux 41.1 (with NMS 41.2); FCOS without NMS 19.0, with NMS 40.9; removing NMS from one-to-many FCOS drops 40.5 -> 12.1. CrowdHuman AP50 89.1 vs 86.1.
- compute: 3DMF adds little; 36 epochs vs 500 for DETR.
- code: https://github.com/Megvii-BaseDetection/DeFCN
- fit_to_us: reuse_idea - effort S - class probability x geometric quality, argmax per GT, is a matching-free rule for "which ray sample is the positive"; 3D max filter = learned NMS over ray and ring neighbours.
- compat_needs: dense score maps; a geometric quality (IoU analogue: distance).
- compat_conflicts: labels depend on current predictions (non-stationary); code is Detectron2-based.
- tried_before: none
- verified: checked

### CARD MH20 - OneNet (classification cost makes one-to-one assignment NMS-free; min-cost without bipartite matching)
- papers: What Makes for End-to-End Object Detection? (Sun et al., ICML 2021) https://arxiv.org/abs/2012.05780 (read via ar5iv)
- depth: M
- stage: L
- idea: Location cost alone yields redundant predictions; adding a classification term to the matching cost gives one prediction per object so NMS is unnecessary. The assignment can be "MinCost": each GT takes its minimum-cost sample, no bipartite matching.
- formulation: C = lambda_cls * C_cls + C_loc, C_loc = lambda_iou C_iou + lambda_L1 C_L1 (lambda_iou = 0 for point candidates); one-to-one: each GT gets the single sample with minimum cost (usually bipartite matching to avoid conflicts). Appendix D: MinCost matched bipartite matching on COCO for RetinaNet and FCOS but failed on CrowdHuman. Coefficient values not given.
- input / output: dense-detector heads (RetinaNet, FCOS, CenterNet) and set predictors.
- training: regime `matching_free` (MinCost) or `hungarian_set`.
- differentiable: yes (labels constant).
- sensors_data: COCO, CrowdHuman.
- results: Table 2 AP (with NMS in brackets): RetinaNet location-only 33.6 (36.8), + classification 36.0 (36.2), + predicted location 37.5 (37.5); FCOS 34.9 (37.7) -> 38.9 (38.9); CenterNet 23.4 (32.0) -> 34.9 (34.8); DETR 12.6 -> 40.0, Deformable DETR 12.0 -> 44.0, Sparse R-CNN 20.1 -> 45.0 when the classification cost is added.
- compute: not stated.
- code: not checked
- fit_to_us: reuse_idea - effort S - direct evidence that argmin-per-GT labelling replaces Hungarian in sparse scenes; the CrowdHuman failure is the warning for parallel lanes and junction mouths.
- compat_needs: a cost combining score and geometry.
- compat_conflicts: crowded scenes need conflict resolution (bipartite).
- tried_before: none
- verified: checked

### CARD MH21 - YOLOv10 consistent dual assignments (one-to-many head for training, top-1 one-to-one head for NMS-free inference)
- papers: YOLOv10: Real-Time End-to-End Object Detection (Wang et al., 2024) https://arxiv.org/abs/2405.14458 (arXiv HTML)
- depth: M
- stage: L, T
- idea: Two heads on one backbone: a one-to-many head (rich supervision) and a one-to-one head with the same structure and objectives; both use one matching metric with the same exponents, so the best one-to-many positive is the one-to-one top-1. Only the one-to-one head is used at inference.
- formulation: m(alpha, beta) = s * p^alpha * IoU(b_hat, b)^beta (s spatial prior, p class score); consistent setting alpha_o2o = r * alpha_o2m, beta_o2o = r * beta_o2m with r = 1; supervision gap measured by the 1-Wasserstein distance between classification targets; one-to-one assignment is top-1 selection (not Hungarian, "same performance as Hungarian" with less training time).
- input / output: detector heads; boxes.
- training: joint training; regime `matching_free`.
- differentiable: yes.
- sensors_data: COCO (images).
- results: YOLOv10-S COCO val: one-to-many only (with NMS) 44.9 AP, 7.07 ms; one-to-one only 43.4 AP, 2.44 ms; dual 44.3 AP, 2.44 ms; mismatched metrics 42.7 (beta = 2.0) or 44.2 (beta = 4.0, alpha = 0.5); a gap to NMS-based training remains for small models.
- compute: no inference overhead.
- code: not checked
- fit_to_us: reuse_idea - effort S - "dense head trains the features, sparse head picks one per object with the same metric" is the NMS-free pattern for a ray fan.
- compat_needs: two heads sharing features and one scalar matching metric.
- compat_conflicts: designed for box detection.
- tried_before: none
- verified: checked

### CARD MH22 - SimOTA / OTA (dynamic-k positives per ground truth, no Hungarian)
- papers: YOLOX: Exceeding YOLO Series in 2021 (Ge et al., 2021) https://arxiv.org/abs/2107.08430 (ar5iv, SimOTA described briefly; the dynamic-k rule itself is in the OTA paper, not read)
- depth: M (limited)
- stage: L
- idea: Positives per GT are the k lowest-cost predictions inside a centre region, with k varying per GT; SimOTA replaces the Sinkhorn-Knopp solver of OTA by this dynamic top-k.
- formulation: c_ij = L_cls + lambda * L_reg (lambda value not given on the page); the page does not give the rule for k (refers to OTA), the conflict rule or the centre radius.
- input / output: dense detector heads; boxes.
- training: regime `matching_free` (many-to-one by dynamic k).
- differentiable: yes (labels constant).
- sensors_data: COCO.
- results: "+multi positives" baseline 45.0 AP -> SimOTA 47.3 AP (+2.3); OTA Sinkhorn-Knopp adds about 25 % training time.
- compute: SimOTA cheaper than OTA, no number.
- code: not checked
- fit_to_us: reuse_idea - effort S - a variable number of positive children per tip depending on quality.
- compat_needs: cost per pair.
- compat_conflicts: needs NMS at inference (many positives).
- tried_before: none
- verified: partly

### CARD MH23 - Group DETR (K groups of queries, each matched one-to-one; one group at inference)
- papers: Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment (Chen et al., ICCV 2023) https://arxiv.org/abs/2207.13085 (read via ar5iv)
- depth: M
- stage: L, T
- idea: Keep Hungarian one-to-one matching, but run K independent groups of N queries through the same decoder (self-attention restricted to the group); each GT gets K positives, one per group. Only group 1 is used at inference. Naive one-to-many assignment inside one group makes duplicates.
- formulation: L = (1/K) * sum_k sum_n l(y_sigma_k(n), ybar_kn), sigma_k = Hungarian matching of group k; K = 11 by default (gain saturates); weights shared; separate self-attention per group is essential (group-wise one-to-many alone 34.8 mAP; with separate self-attention 37.6; naive one-to-many 8.4 vs 32.6 baseline, Conditional DETR 12 epochs).
- input / output: DETR-family decoders; sets of boxes + classes.
- training: 12 / 36 / 50 epochs on COCO; regime `hungarian_set` per group.
- differentiable: yes.
- sensors_data: COCO val2017.
- results: 12-epoch mAP baseline -> Group DETR: Conditional DETR 32.6 -> 37.6; DAB-DETR 35.2 -> 39.1; DN-DETR 38.6 -> 40.6; DAB-Deformable 44.2 -> 45.7; DINO 4-scale 49.4 -> 50.1; at 50 epochs DINO +0.4.
- compute: training only: about 1.2 GB (Conditional) / 1.7 GB (DAB) more memory, about 5 min more per epoch.
- code: https://github.com/Atten4Vis/GroupDETR (named in a search result)
- fit_to_us: not_suitable as published (Hungarian in every group) - effort - ; the transferable idea is K positives per target in training, one query set at inference.
- compat_needs: per-group Hungarian; independent self-attention per group.
- compat_conflicts: Hungarian is excluded by project rule; a tree has no query set.
- tried_before: E31 / E32 (matched sparse decoder) dropped
- verified: checked

### CARD MH24 - H-DETR hybrid matching (one-to-one branch + one-to-many branch with repeated GT)
- papers: DETRs with Hybrid Matching (Jia et al., CVPR 2023) https://arxiv.org/abs/2207.13080 (read via ar5iv)
- depth: M
- stage: L, T
- idea: Two query groups share the decoder: N = 300 queries one-to-one, T = 1500 queries one-to-many by repeating every GT K = 6 times and applying Hungarian matching to the repeated targets; masked self-attention separates the groups; only the one-to-one branch is used at inference.
- formulation: total = lambda * L_one2many + L_one2one, lambda = 1 (0.1: 47.8, 1: 48.7, 5: 48.3 AP); K: 0 -> 47.0, 1 -> 46.4, 3 -> 48.1, 6 -> 48.6, 8 -> 48.6 (gain only for K > 3); T: 300 -> 47.8 ... 1500 -> 48.7. Variants hybrid epoch (one-to-many in the first 2/3 of epochs, K~ = 10) and hybrid layer (first 4 decoder layers one-to-many).
- input / output: Deformable-DETR-like set predictors; also pose (PETR), 3D detection (PETRv2), tracking.
- training: COCO 12 / 24 / 36 epochs; Hungarian per branch; regime `hungarian_set`.
- differentiable: yes.
- sensors_data: COCO, LVIS, nuScenes (PETRv2 camera 3D, NDS 50.68 -> 52.38).
- results: Deformable-DETR R50 43.7 / 46.4 / 46.8 (12 / 24 / 36 ep) -> hybrid branch 45.9 / 47.6 / 48.0; COCO test Swin-L 59.4 AP; one-to-many alone with NMS 49.4 / 50.2 / 48.8 (5.3 FPS) vs hybrid one-to-one 48.7 / 49.9 / 50.0 (6.7 FPS).
- compute: training time +23.1 % (R50) / +6.4 % (Swin-L), memory 5.5 -> 7.7 GB (R50); no inference cost.
- code: not checked
- fit_to_us: reuse_idea - effort M - training-only redundancy: extra children per tip in training (K repeats) that are dropped at test time.
- compat_needs: Hungarian on repeated GT (replace by nearest assignment).
- compat_conflicts: the no-Hungarian rule; extra children multiply tree size.
- tried_before: none
- verified: checked

### CARD MH25 - Co-DETR (auxiliary dense one-to-many heads + positive queries without matching)
- papers: DETRs with Collaborative Hybrid Assignments Training (Zong, Song, Liu, ICCV 2023) https://arxiv.org/abs/2211.12860 (read via ar5iv)
- depth: M
- stage: L, T
- idea: Attach conventional one-to-many heads (ATSS, Faster R-CNN RPN / RoI, ...) to the DETR encoder features during training, and feed their positive anchors / proposals as extra decoder query groups where each query is supervised directly by the GT assigned to it (no Hungarian). Everything auxiliary is dropped at inference.
- formulation: loss = sum_layers L_one2one + lambda1 * L_aux-decoder-queries + lambda2 * sum_heads L_encoder-head; lambda1 = 1.0, lambda2 = 2.0, K = 2 heads (ATSS + Faster R-CNN) or 1; positive query = linear(PE(box)) + linear(encoder feature at the location); about 8.8 (ATSS) and 18.7 (Faster R-CNN) positives per GT. Argument: one-to-one matching supervises the encoder sparsely and Hungarian matching is unstable.
- input / output: multi-scale encoder features -> boxes.
- training: COCO 12 / 36 epochs; regime `hungarian_set` (main) + `matching_free` (auxiliary queries).
- differentiable: yes.
- sensors_data: COCO.
- results: Deformable-DETR 12 ep R50 37.1 -> 42.9 AP (K = 2); 36 ep 43.3 -> 46.8 (K = 1); DINO-Deformable 12 ep 49.4 -> 51.2; Swin-L DINO 58.5 -> 59.5. K = 3: 49.5, K = 6: 48.9 (conflicts between diverse heads).
- compute: GPU hours 70 -> 86 (K = 1) -> 120 (K = 2); memory 12.8 -> 13.9 -> 14.4 GB; no inference overhead.
- code: not checked
- fit_to_us: reuse_idea - effort M - dense auxiliary supervision on the shared features and positive queries with their own assigned GT are matching-free; our heat map + tracer already follows the dense-head pattern (E31 / E32).
- compat_needs: dense auxiliary heads on the same features; a positive rule.
- compat_conflicts: published code is mmdet-style (mmdet / mmcv not in our environment).
- tried_before: E31 / E32 (matched sparse decoder dropped, grid kept)
- verified: checked

### CARD MH26 - DAC-DETR (auxiliary decoder without self-attention, one-to-many by score threshold)
- papers: DAC-DETR: Divide the Attention Layers and Conquer (Hu, Sun, Wang, Yang, NeurIPS 2023) https://papers.nips.cc/paper_files/paper/2023/hash/edd0d433f8a1a51aa11237a6543fc280-Abstract-Conference.html (method read from a text copy of the paper at object.cloud.sdsc.edu)
- depth: M
- stage: L, T
- idea: Cross-attention gathers several queries on one object, self-attention disperses them. A training-only auxiliary decoder without self-attention (shared cross-attention, FFN, queries, head) is trained one-to-many to strengthen the gathering; the main decoder keeps Hungarian one-to-one. The auxiliary decoder is dropped at inference.
- formulation: per GT, score M = p_q(c_hat) + IoU(b_q, b_hat) over all queries; positive if M above a threshold t and among the top-k (t and k not given in the text read); L = L_O (one-to-one) + L_C (focal + GIoU); Hungarian with repeated GT in the auxiliary branch is worse (45.5 vs 47.1 AP).
- input / output: DETR decoders; boxes.
- training: COCO 12 epochs; regime `hungarian_set` (main) + `matching_free` (aux).
- differentiable: yes.
- sensors_data: COCO.
- results: Deformable-DETR R50 12 ep 43.7 -> 47.1 AP (+3.4); DINO 49.0 -> 50.0; with the Align-DETR loss 50.9 (R50) / 58.1 (Swin-L). Ablations: frozen cross-attention in main decoder 45.9; self-attention in the auxiliary decoder 43.1; one-to-one in the auxiliary decoder 43.5.
- compute: overhead not reported.
- code: https://github.com/huzhengdongcs/DAC-DETR ("will be released" per the abstract)
- fit_to_us: reuse_idea - effort S - gather (cross-attention to features) versus separate (self-attention among siblings) are two roles; our verification and sibling-attention stages play them.
- compat_needs: a DETR-style decoder.
- compat_conflicts: none for the idea; code reuse needs the DETR stack.
- tried_before: none
- verified: checked

### CARD MH27 - DN-DETR (query denoising: noised GT queries with known assignment)
- papers: DN-DETR: Accelerate DETR Training by Introducing Query DeNoising (Li et al., CVPR 2022) https://arxiv.org/abs/2203.01305 (read via ar5iv)
- depth: M
- stage: L, T
- idea: Hungarian matching is unstable early in training (the same query is matched to different objects across epochs). Add extra decoder queries built from noised GT boxes and labels; each reconstructs its own GT, so no matching is needed; they are removed at inference.
- formulation: box noise: centre shift |dx| < lambda1 * w/2, |dy| < lambda1 * h/2; scale in [(1 - lambda2) w, (1 + lambda2) w]; label flip ratio gamma; defaults lambda1 = lambda2 = 0.4, gamma = 0.2, 5 denoising groups; attention mask: matching queries cannot see denoising queries and denoising groups cannot see each other (without it performance degrades); instability index = number of query-to-target assignment changes between consecutive epochs.
- input / output: DAB-DETR / Deformable DETR decoders.
- training: 50 epochs; regime `denoising` + `hungarian_set`.
- differentiable: yes.
- sensors_data: COCO.
- results: DAB-DETR R50 50 ep 42.2 -> 44.1 (groups 0 / 1 / 5: 42.2 / 43.4 / 44.1); DC5 12 ep 38.0 -> 41.7; DN-Deformable 48.6 vs 46.9; DINO +1.4 at 12 ep.
- compute: GFLOPs 94.4 -> 94.5 - 94.6; epoch time 51.1 -> 57.7 min.
- code: not checked
- fit_to_us: reuse_idea - effort S - "noised GT as hypotheses with a known target" is a matching-free training signal for the refine / rescore stages and for a recursion that starts from the noised previous set (cf. stream query denoising in the earlier review).
- compat_needs: a mask separating denoising queries from hypothesis queries.
- compat_conflicts: trains refinement and scoring, not generation of new hypotheses.
- tried_before: none
- verified: checked

### CARD MH28 - MapTRv2 auxiliary one-to-many branch (map elements on nuScenes)
- papers: MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction (Liao et al., 2023) https://arxiv.org/abs/2308.05736 (arXiv HTML)
- depth: M
- stage: L, T
- idea: Add a training-only group of T = 300 extra instance queries sharing the decoder; GT elements are repeated K = 6 times (padded to size T) and the same hierarchical bipartite matching is applied, so one GT can match 6 predictions.
- formulation: loss weight 1; base matching: instance-level Hungarian (focal class cost + Manhattan point cost) then point-level permutation with the lowest Manhattan sum (2 permutations for undirected polylines, 1 for directed centrelines).
- input / output: BEV features -> vectorised map elements.
- training: nuScenes 24 epochs; regime `hungarian_set`.
- differentiable: yes.
- sensors_data: nuScenes (camera, optionally lidar).
- results: nuScenes val R50 24 ep: mAP 57.6 (no branch) -> 61.5 (K = 6); loss weight 0.1 -> 58.5, 1.0 -> 61.5, 2.0 -> 59.9; the three dense auxiliary losses (depth, perspective-view and BEV segmentation) together 56.6 -> 61.5 (+4.9) as read from the page.
- compute: not stated.
- code: https://github.com/hustvl/MapTR (named; licence not stated on the page)
- fit_to_us: baseline - effort M - proves the "dense positives in training only" trick in the nuScenes map domain.
- compat_needs: Hungarian.
- compat_conflicts: Hungarian.
- tried_before: E31 / E32 (matched sparse decoder dropped)
- verified: checked

### CARD MH29 - Align-DETR (IoU-aware BCE target for the score; rank-weighted k positives)
- papers: Align-DETR: Enhancing End-to-end Object Detection with Aligned Loss (Cai et al., 2023) https://arxiv.org/abs/2304.07527 (read via ar5iv)
- depth: M
- stage: L
- idea: Train the classification score toward a joint quality metric so that confidence reflects localisation quality (ranking by score then equals ranking by quality); positives use a soft target, negatives focal loss; intermediate layers take k = 4 matched candidates per GT with rank-decaying weights.
- formulation: target t_c = exp(-r / tau) * q, q = p^alpha * u^(1 - alpha), r = candidate rank, alpha = 0.25, tau = 1.5; regression weight exp(-r / tau); k = 4 in intermediate layers, k = 1 in the last.
- input / output: DETR decoders.
- training: COCO 12 / 24 epochs; regime `hungarian_set` (modified).
- differentiable: yes.
- sensors_data: COCO.
- results: Align-DETR 50.5 AP (12 ep) / 51.7 (24 ep); DINO 1x 49.0; H-DETR baseline 48.7 -> 49.3. The paper does not frame it as calibration.
- compute: not stated.
- code: not checked
- fit_to_us: reuse_idea - effort S - template for the validity score target: p^alpha * closeness^(1 - alpha) with closeness from point-to-polyline distance.
- compat_needs: a closeness measure in [0, 1].
- compat_conflicts: none.
- tried_before: none
- verified: checked

### CARD MH30 - One-to-many assignment plus NMS beats Hungarian (DETA) and one-to-several lane detection (O2SFormer)
- papers: NMS Strikes Back (Ouyang-Zhang, Cho, Zhou, Krahenbuhl, 2022) https://arxiv.org/abs/2212.06137 (abstract); End-to-End Lane detection with One-to-Several Transformer (Zhou, Zhou, 2023) https://arxiv.org/abs/2305.00675 (search summary)
- depth: A
- stage: L
- idea: DETA: IoU-based one-to-many assignment with NMS outperforms one-to-one Hungarian matching in the same transformer. O2SFormer: lane detection mixing one-to-one and one-to-several assignment with a layer-wise soft label and dynamic anchor-based positional queries.
- formulation: not read.
- input / output: not read (lane anchors in O2SFormer).
- training: not read.
- differentiable: not read.
- sensors_data: COCO (DETA); CULane (O2SFormer).
- results: DETA up to +2.5 mAP over one-to-one, Deformable-DETR 50.2 mAP at 12 epochs (abstract); O2SFormer R50 77.83 % F1 on CULane, converges 12.5x faster than DETR with R18 (search summary).
- compute: not stated.
- code: https://github.com/zkyseu/O2SFormer (named on a listing page, not opened)
- fit_to_us: baseline - effort S - supports keeping dense positives + suppression as the strong baseline against any learned selection.
- compat_needs: NMS at inference.
- compat_conflicts: NMS is not differentiable.
- tried_before: none
- verified: partly

### CARD MH31 - Relation Networks for duplicate removal (learned NMS from rank embedding + relation module)
- papers: Relation Networks for Object Detection (Hu, Gu, Zhang, Dai, Wei, CVPR 2018) https://arxiv.org/abs/1711.11575 (read via ar5iv)
- depth: M
- stage: S4, S5
- idea: A relation module lets each detection attend to all others with appearance and geometry weights; the duplicate-removal network takes score rank, box and feature per detection and outputs s1 in [0,1]; final score = s0 * s1; trained end to end.
- formulation: f_R(n) = sum_m w^{mn} (W_V f_A^m); w^{mn} = w_G^{mn} exp(w_A^{mn}) / sum_k w_G^{kn} exp(w_A^{kn}); w_A = dot(W_K f_A^m, W_Q f_A^n) / sqrt(d_k); w_G = max(0, W_G E_G(f_G^m, f_G^n)) with a sinusoidal embedding of the log-ratio box offsets; residual update; N_r = 16, d_k = 64, d_g = 64. Duplicate net: rank (1..N) embedded to 128-d (works better than the raw score) + projected feature, sigmoid output. Label: per GT, detections with IoU >= eta; the highest-scoring is 1, others 0; eta in {0.5, ..., 0.9}, one output per threshold, averaged; loss BCE on s0 * s1; labels change during training because they depend on the detector's current scores.
- input / output: set of detections with features and boxes -> rescored set.
- training: equal-weight sum of RPN, recognition and duplicate losses; regime `matching_free` (argmax per GT among current detections).
- differentiable: yes (gradient flows to head and backbone; labels constant).
- sensors_data: COCO (Faster R-CNN R50).
- results: instance recognition 29.6 -> 31.9 mAP; duplicate removal: NMS 29.6, SoftNMS 30.2, learned 30.5 separate / 31.0 end to end.
- compute: duplicate net about 0.33 M parameters, 0.3 GFLOPs, about 2 ms (N = 100, Titan X); a full relation module about 3 M parameters.
- code: not checked
- fit_to_us: reimplement - effort S - sibling / ancestor attention with rank embedding and "best one per GT is positive, others duplicates" is our stage C label, matching-free.
- compat_needs: scores sorted to ranks; geometry of the set.
- compat_conflicts: label depends on current scores.
- tried_before: none
- verified: checked

### CARD MH32 - GossipNet / Tnet (learned NMS on boxes and scores only)
- papers: Learning Non-Maximum Suppression (Hosang, Benenson, Schiele, CVPR 2017) https://arxiv.org/abs/1705.02950 (read via ar5iv); predecessor: A convnet for non-maximum suppression https://arxiv.org/abs/1511.06437 (read; Tnet)
- depth: M
- stage: S4
- idea: A pure NMS network takes only boxes and scores; each detection has a 128-d feature updated in blocks that pair it with neighbours of IoU > 0.2, apply pairwise FC layers, max-pool over neighbours and add residually (default 16 blocks); one score per detection.
- formulation: labels: detections are matched to GT in descending confidence as in the benchmark, matched = positive, unmatched (incl. duplicates) = negative; weighted logistic loss; Tnet (predecessor) is a fully convolutional net on a 2-D grid fed with GreedyNMS score maps at several thresholds.
- input / output: detection set -> per-detection keep score.
- training: Adam 2 M iterations on COCO; not trained jointly with the detector.
- differentiable: yes inside the NMS net; not end to end with the detector.
- sensors_data: PETS, COCO persons (8-block net).
- results: COCO persons AP0.5:0.95 36.9 vs 35.6 (GreedyNMS > 0.5) on minival; PETS +4.8 AP overall, +7.3 on highly occluded; Tnet oMNIST AR 86.0 vs 60.2.
- compute: about 14 ms per image for 67 detections on a K40m (unoptimised TensorFlow).
- code: not checked
- fit_to_us: reuse_idea - effort S - a set-to-set rescorer working on geometry only, usable on the b x M candidates before the beam cut.
- compat_needs: neighbours by IoU analogue (distance).
- compat_conflicts: needs large data; gains modest (about 1 AP on COCO).
- tried_before: none
- verified: checked

### CARD MH33 - End-to-end single-shot detector with graph-based learnable duplicate removal (Ding et al.)
- papers: End-to-End Single Shot Detector using Graph-based Learnable Duplicate Removal (Ding, Rehder, Schneider, Cordts, Gall, GCPR 2022) https://pages.iai.uni-bonn.de/gall_juergen/download/nonmaximumsuppression_gcpr22.pdf (listing and abstract only; PDF text not decodable)
- depth: A
- stage: S4, S5
- idea: A GNN over detections refines scores and boxes from image features, trained end to end with an SSD detector; a pre-filtering head cuts the number of proposals before the graph.
- formulation: not read.
- input / output: SSD proposals -> rescored detections.
- training: end to end with the detector (abstract).
- differentiable: yes (abstract).
- sensors_data: MS-COCO, KITTI.
- results: "improves accuracy compared with other duplicate removal methods at significantly lower inference time" (abstract); no numbers seen.
- compute: not stated.
- code: none found
- fit_to_us: reuse_idea - effort S - same role as MH31; pre-filter + graph rescoring maps onto "take the best 3 x M candidates, then a GNN over siblings".
- compat_needs: not read.
- compat_conflicts: not read.
- tried_before: none
- verified: partly

### CARD MH34 - SOFT top-k by entropic optimal transport (differentiable top-k, soft beam search)
- papers: Differentiable Top-k with Optimal Transport (Xie et al., NeurIPS 2020) https://arxiv.org/abs/2002.06504 (read via ar5iv; last 22 k characters not read)
- depth: M
- stage: S4
- idea: Top-k is cast as transporting n scores to two anchors {0, 1} with masses k/n and (n - k)/n; the entropic plan gives a smooth indicator of the top-k set whose gradient is computed implicitly.
- formulation: mu = 1/n over the scores; nu = (k/n, (n - k)/n) over {0, 1}; cost C_i1 = x_i^2, C_i2 = (x_i - 1)^2; Sinkhorn: G = exp(-C/eps), iterate p = mu / (G q), q = nu / (G^T p), plan = diag(p) G diag(q); smoothed indicator A = n * plan * [0, 1]^T. Backward: implicit differentiation of the KKT conditions (closed form in the appendix), no unrolling. Error bound ~ eps (ln n + ln 2) / (n * gap between the k-th and (k+1)-th score). eps used 1e-3 (MNIST kNN), 1e-5 (CIFAR-10 kNN), 1e-1 (beam search); iteration count not extracted. Cost O(n) per iteration (O(k n) for sorted top-k).
- input / output: vector of n scores -> soft indicator (or soft sorted top-k).
- training: inside kNN classifiers and inside beam search of machine translation (soft top-k replaces hard top-k in training, mixed with teacher forcing, rho = 0.8, beam 5).
- differentiable: yes (smooth, Jacobian has a nonzero entry for any input); bias from eps.
- sensors_data: MNIST, CIFAR-10, WMT14 En-Fr, WMT16 En-De.
- results: kNN CIFAR-10 92.6 % (k-times softmax 92.2, NeuralSort 90.7); MNIST 99.4 (NeuralSort 99.5); WMT14 En-Fr beam search 36.27 BLEU vs 35.38 reproduced baseline; WMT16 En-De top-k attention 37.30 vs 36.54 soft attention.
- compute: O(n) per Sinkhorn iteration; implicit backward avoids storing iterations.
- code: not checked
- fit_to_us: reimplement - effort S - ten lines on the M x b scores; gradient reaches all candidates including dropped ones; the soft indicator can weight node contributions in the next level.
- compat_needs: a scalar score per candidate; a loss that uses the soft weights (otherwise nothing is learned from them).
- compat_conflicts: dropped nodes stay partly in the graph during training, inference needs a hard top-M (train / test mismatch for eps > 0); eps trades bias against gradient smoothness.
- tried_before: none
- verified: checked

### CARD MH35 - sparsemax and alpha-entmax (sparse normalised weights with exact zeros; exact beam search)
- papers: From Softmax to Sparsemax (Martins, Astudillo, ICML 2016) https://arxiv.org/abs/1602.02068; Sparse Sequence-to-Sequence Models (Peters, Niculae, Martins, ACL 2019) https://arxiv.org/abs/1905.05702 (both read via ar5iv)
- depth: M
- stage: S4, L
- idea: Replace softmax by the Euclidean projection onto the simplex (sparsemax) or its alpha-entmax generalisation: outputs have exact zeros, so a distribution over children has a small support; with a sparse output the set of positive-probability hypotheses can be enumerated and beam search becomes exact when the beam covers the support.
- formulation: sparsemax p_i = max(0, z_i - tau(z)), tau from a sort (O(K log K)); Jacobian diag(s) - s s^T / |S(z)| (zero rows outside the support); sparsemax loss convex with gradient -q + sparsemax(z), zero once the margin is >= 1. alpha-entmax: p_i = [(alpha - 1) z_i - tau]_+^(1/(alpha - 1)); alpha = 2 is sparsemax, alpha -> 1 softmax; Fenchel-Young loss gradient p - e_y; exact sort algorithm for alpha = 1.5, bisection for general alpha; attention Jacobian diag(s) - s s^T / ||s||_1 with s_i = p_i^(2 - alpha). Output-loss alpha best near 1.33; alpha > 2 did not help. A loss for a distribution target y (two-hot for a fork) is the same Fenchel-Young form with p - y (from memory, not on the pages read).
- input / output: score vector -> sparse probability vector.
- training: replaces softmax cross-entropy; alpha is a hyper-parameter.
- differentiable: yes; entries outside the support get zero gradient.
- sensors_data: SIGMORPHON 2018 inflection, six NMT pairs, SNLI attention, multi-label benchmarks.
- results: beam search exact for 87 % (high-resource) and 79 % (medium) validation sequences in inflection; 1.5-entmax keeps about 16 of 17,993 target types in NMT; BLEU 1.5-entmax vs softmax de->en 26.17 vs 25.70, ro->en 30.15 vs 29.12; sparsemax below softmax on most NMT pairs; SNLI sparse attention 82.20 vs 82.08 soft vs 80.99 none.
- compute: de->en about 10,500 / 13,000 / 14,500 target words per second (bisection / exact 1.5-entmax / softmax, GTX 1080).
- code: not checked
- fit_to_us: reuse_idea - effort S - normalised competition over the b children of a tip with exact zeros for children that are not needed (a differentiable soft NMS).
- compat_needs: scores that compete (sum to 1) within a tip.
- compat_conflicts: one winner takes the mass whereas a fork has two true children (needs a two-hot target); an item that is wrongly zero gets no gradient and cannot recover; no demonstration of entmax inside a training-time beam.
- tried_before: none
- verified: checked

### CARD MH36 - Differentiable ranking and sorting by optimal transport (soft rank, soft sort, soft quantile)
- papers: Differentiable Ranking and Sorting using Optimal Transport (Cuturi, Teboul, Vert, NeurIPS 2019) https://arxiv.org/abs/1905.11885 (read via ar5iv)
- depth: M
- stage: S4
- idea: Sorting is an optimal assignment between the input values and a target measure on increasing anchor values; entropic regularisation gives a dense differentiable plan, from which soft rank and soft sort operators (and a soft quantile with a 3-point target measure) follow.
- formulation: Sinkhorn on the n x m cost between inputs and m anchors; soft ranks / sorts from the plan; gradients by autodiff through the Sinkhorn iterations (not implicit); cost scales with iterations x n x m; top-k classification uses a ReLU-type loss on the soft rank, J_k(u) = max(0, u - k + 1), experiments only for k = 1; suggested eps 1e-2 or 1e-3 for inputs in [0, 1]; eps -> 0 recovers hard rank / sort; small eps gives vanishing gradients, large eps collapses outputs to averages.
- input / output: vector of values -> soft ranks / sorted values.
- training: CIFAR-10 / CIFAR-100 soft top-1 loss (CNN, ResNet18), MNIST sorting (on par with NeuralSort), least-quantile regression.
- differentiable: yes (bias from eps).
- sensors_data: CIFAR, MNIST.
- results: no absolute accuracy numbers in the text read (figures only); MNIST sorting precision slightly ahead of NeuralSort (Table 1).
- compute: O(iterations * n * m).
- code: not checked
- fit_to_us: baseline - effort S - alternative to MH34 for soft top-M of the candidate scores; same bias trade-off.
- compat_needs: scalar scores.
- compat_conflicts: no soft top-k defined by the paper itself (only k = 1 loss).
- tried_before: none
- verified: checked

### CARD MH37 - Differentiable perturbed optimizers (noise-smoothed argmax / top-k with Fenchel-Young loss)
- papers: Learning with Differentiable Perturbed Optimizers (Berthet et al., NeurIPS 2020) https://arxiv.org/abs/2002.08676 (read via ar5iv)
- depth: M
- stage: S4, L
- idea: Replace a non-differentiable solver (argmax, top-k, shortest path) by its expectation under random perturbation of the input; the Jacobian is obtained by integration by parts and estimated with Monte Carlo; a Fenchel-Young loss needs only the perturbed maximiser.
- formulation: y_eps(theta) = E[y*(theta + eps Z)], Z density proportional to exp(-nu(z)); Jacobian = E[y*(theta + eps Z) grad nu(Z)^T] / eps; Monte Carlo over M samples; L_eps(theta; y) = F_eps(theta) + eps Omega(y) - <theta, y>, gradient y_eps(theta) - y. Smooth, strictly convex F_eps, y_eps interior, gradient never zero; error bounds proportional to eps; M = 1, eps = 1 for shortest paths, M in {1, 1000} on CIFAR-10. Top-k appears only as an example polytope (Appendix B); no top-k experiments.
- input / output: score vector -> smoothed indicator / path.
- training: CIFAR-10 (600 epochs, batch 32), label ranking, Warcraft shortest paths (ResNet18, 50 epochs).
- differentiable: yes (smoothing bias eps).
- sensors_data: CIFAR-10, Warcraft terrain, 21 label-ranking sets.
- results: perturbed FY loss competitive with cross-entropy on CIFAR-10; shortest-path numbers not extracted.
- compute: M solver calls per gradient (parallelisable).
- code: not checked
- fit_to_us: reuse_idea - effort S - noise-based top-M with a task-agnostic gradient, no Sinkhorn; M samples of a size-30 top-k are cheap.
- compat_needs: a solver that can be called on perturbed scores.
- compat_conflicts: variance with small M; no published top-k experiment.
- tried_before: none
- verified: checked

### CARD MH38 - Subset-sampling relaxations: relaxed top-k (Xie-Ermon) and SIMPLE (exact k-subset forward, marginals backward)
- papers: Reparameterizable Subset Sampling via Continuous Relaxations (Xie, Ermon, IJCAI 2019) https://arxiv.org/abs/1901.10517; SIMPLE: A Gradient Estimator for k-Subset Sampling (Ahmed, Zeng, Niepert, Van den Broeck, ICLR 2023) https://arxiv.org/abs/2210.01941 (both read via ar5iv)
- depth: M
- stage: S4
- idea: Draw k DISTINCT items with gradient. Xie-Ermon: Gumbel noise plus k successive softmaxes with the chosen item suppressed by log(1 - a). SIMPLE: sample an exact discrete k-subset in the forward pass (dynamic programming for the exactly-k distribution, O(n k)); in the backward pass use the Jacobian of the conditional marginals p(z_i = 1 | sum z = k).
- formulation: relaxed top-k temperature t > 0; exact samples as t -> 0 (Prop. 1); ordering of weights preserved for t >= 1; complexity O(k m). SIMPLE: marginals are partial derivatives of the log-probability of the k-constraint, so autodiff computes them; for k = 1 it resembles straight-through Gumbel-softmax but differentiates through unperturbed marginals; vectorised O(log k log n) with divide and conquer.
- input / output: n logits -> k-hot vector.
- training: plugged into feature selection and VAEs.
- differentiable: partial - forward exact discrete (SIMPLE) / relaxed (Xie-Ermon); backward biased but low variance.
- sensors_data: IMDB, MNIST / Fashion-MNIST / CIFAR-10 kNN; BeerAdvocate, Taste, discrete VAE MNIST.
- results: Xie-Ermon IMDB-word post-hoc accuracy 90.7 -> 91.7 % (k = 10), kNN MNIST 99.3 % (NeuralSort 99.4), forward pass 0.110 s vs 3.694 s NeuralSort at m = 5000, k = 5. SIMPLE BeerAdvocate Aroma (k = 5) MSE 2.27 vs SoftSub 2.57, I-MLE 2.62, L2X 5.75; Taste (k = 10) 2.11 vs 2.18 / 2.38 / 6.92; lower bias and variance than score function, SoftSub, I-MLE, ST Gumbel-softmax on a 5-of-10 synthetic test.
- compute: O(n k) for the DP.
- code: not checked
- fit_to_us: reuse_idea - effort S - picking exactly M (or b) children with an unbiased-in-forward, low-variance gradient; both fit M x b <= 100 candidates.
- compat_needs: independent logits with an exactly-k constraint.
- compat_conflicts: the distribution is over subsets, it does not know geometric redundancy between picked nodes.
- tried_before: none
- verified: checked

### CARD MH39 - Stochastic Beams and the Gumbel-top-k trick (sampling without replacement through a tree)
- papers: Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement (Kool, van Hoof, Welling, ICML 2019) https://arxiv.org/abs/1903.06059 (read via ar5iv)
- depth: M
- stage: S4
- idea: Gumbel-perturbed log-probabilities, top-k = k samples without replacement. For tree models the same result is obtained inside beam search by keeping the k best perturbed partial scores per level; a child's perturbed score is sampled conditionally on its parent so the tree stays consistent.
- formulation: G_i = phi_i + Gumbel(0); node G_S ~ Gumbel(phi_S); child score G~ = -log( exp(-G_parent) - exp(-Z) + exp(-G_child) ), Z = max over children; expand the k nodes with highest perturbed score per level (contains the ancestors of the top-k leaves). Estimators: weights p(y)/q(y) with inclusion probability q = 1 - exp(-exp(phi - kappa)), kappa = (k+1)-th largest perturbed score; self-normalised version biased but consistent and lower variance.
- input / output: autoregressive model with normalised conditional probabilities; k sampled sequences.
- training: none (inference sampler / estimator); no gradient estimator in this paper.
- differentiable: no (top-k of perturbed scores).
- sensors_data: WMT14 en-fr newstest2014.
- results: k = 5, 10, 20 and T 0.1 - 0.8: better mean and maximum BLEU at equal diversity than diverse beam search; estimation variance below Monte Carlo for T < 0.5.
- compute: linear in k and sequence length.
- code: not checked
- fit_to_us: reuse_idea - effort M - stochastic exploration of the beam in training (different nodes survive in different epochs).
- compat_needs: normalised conditional probabilities child | parent.
- compat_conflicts: our validity scores are independent survival products, not normalised conditionals.
- tried_before: none
- verified: checked

### CARD MH40 - Unordered-set estimator (unbiased gradients from k samples without replacement)
- papers: Estimating Gradients for Discrete Random Variables by Sampling without Replacement (Kool, van Hoof, Welling, ICLR 2020) https://arxiv.org/abs/2002.06043 (read via ar5iv; last 29 k characters not read)
- depth: M
- stage: S4, T
- idea: An unbiased estimator of expectations (and policy gradients) over discrete variables from a set of k samples drawn without replacement (Gumbel-top-k / stochastic beam search); a Rao-Blackwellisation of single-sample and importance-weighted estimators.
- formulation: sum over the sampled set of p(s) * R(S^k, s) * f(s), R = leave-one-out ratio (probability of the other samples under the distribution restricted to D minus {s}, divided by the probability of the full set); a second-order leave-one-out ratio gives a built-in baseline with no extra evaluation; for sequences use stochastic beam search; exact set probability in O(2^k) or by numerical integration (naive O(k!)).
- input / output: a discrete policy -> unbiased gradient.
- training: REINFORCE-type; combined with pathwise terms if f depends on theta.
- differentiable: yes in expectation (unbiased score-function gradient), not through the discrete choice.
- sensors_data: Bernoulli toy, categorical VAE (10^2 and 10^20 latent states), TSP with 20 nodes.
- results: lowest or comparable gradient variance in all three; categorical VAE on par with REINFORCE with replacement; TSP better than REINFORCE with replacement, stochastic sum-and-sample and a greedy rollout baseline, on par with a biased risk estimator.
- compute: modest overhead for small k.
- code: not checked
- fit_to_us: reuse_idea - effort M - the only unbiased route found for training a stochastic top-b / top-M selection, at the price of normalised sampling probabilities.
- compat_needs: normalised probabilities of the candidates; an episode-level reward.
- compat_conflicts: policy-gradient variance; not needed if local supervision of scores exists.
- tried_before: none
- verified: checked

### CARD MH41 - Continuous relaxation of beam search (soft top-k beam trained end to end)
- papers: A Continuous Relaxation of Beam Search for End-to-end Training of Neural Sequence Models (Goyal, Dyer, Berg-Kirkpatrick, AAAI 2018) https://arxiv.org/abs/1708.00111 (read via ar5iv)
- depth: M
- stage: S4, T
- idea: Replace the hard top-k of beam search by a soft top-k built from k peaked softmaxes; the beam state is a convex combination of previous states and expected input embeddings, so a task loss can be applied to the beam output. The temperature alpha is annealed up.
- formulation: for rank i = 1..k: p_i = softmax over candidates of -alpha * (s - max_i)^2 with max_i the i-th largest score (iterated max operations); soft backpointer = row sums of p_i; new hidden state = backpointer-weighted average of previous hidden states; next input = expected embedding a_i^T E; cumulative score s_{t+1,i} = sum(s~_t * p_i). Loss: per-step Hamming loss accumulated along the soft beam, final loss = peaked softmax over final beam scores applied to the accumulated loss (soft max-margin variant worse). alpha annealed from non-peaked to peaked (constant 1.0 worse, large constant unstable in pilots); beam size 3, 50 epochs, warm start with cross-entropy.
- input / output: sequence tagger (BiLSTM encoder, LSTM decoder, hidden 64 NER / 512 CCG); k label hypotheses.
- training: regime `free_rollout` through the soft beam.
- differentiable: yes at finite alpha (all candidates get gradient); hard beam recovered for alpha -> inf if no ties.
- sensors_data: German NER (CoNLL 2003), CCG supertagging (CCGbank); no NMT experiment.
- results: CCG tag accuracy: cross-entropy greedy 80.35, cross-entropy hard beam 82.42, soft-beam training 85.78 (soft) / 85.82 (hard beam at test); NER macro F1: 54.92 greedy cross-entropy vs 56.38 / 56.15.
- compute: about k times cross-entropy training; flat objective with huge gradients near change points at high alpha.
- code: not checked
- fit_to_us: reuse_idea - effort M - template for "prune M nodes per level with gradient"; state mixing is only defined for recurrent states, a convex combination of two positions is not a lane point.
- compat_needs: scalar score per candidate; averageable states.
- compat_conflicts: averaging node positions or tokens blends different lanes; cost grows with the number of soft selections.
- tried_before: none
- verified: checked

### CARD MH42 - Fully differentiable beam search decoder (beam as a computation graph, gold paths added to the normaliser)
- papers: A Fully Differentiable Beam Search Decoder (Collobert, Hannun, Synnaeve, ICML 2019) https://arxiv.org/abs/1902.06022 (read via ar5iv)
- depth: M
- stage: S4, T
- idea: Express the decoder (lexicon, transition and word-LM scores, merging of equal hypotheses by logadd, pruning to a beam) as differentiable operations and train the acoustic model through it with a sequence-level criterion; the gold alignments are always included in the normaliser.
- formulation: per hypothesis (lexicon state, LM state, score); equal states merged with logadd; beam keeps the top scores (discrete; no gradient through the top-K decision, gradient through the scores of the kept hypotheses); loss = -( logadd over gold alignments (Forward) - log( beam sum - beam-gold sum + gold-alignment sum ) ) (Eq. 8); too small a beam diverges; beam 500 mostly (1000, 2000 tested).
- input / output: frame token scores -> word sequence.
- training: beam-level criterion; regime `free_rollout` through the beam output.
- differentiable: partial - pruning hard; gradient through kept scores only.
- sensors_data: Wall Street Journal speech.
- results: WSJ test WER: ASG 10 M model beam 8000 5.6; DBD 10 M beam 500 5.9; DBD 7.5 M beam 500 5.3, beam 1000 5.1; acoustic-only ASG 4-gram 11.3 vs DBD 4-gram 9.9.
- compute: epoch 35 / 90 / 180 min at beam 500 / 1000 / 2000; CPU-only criterion; custom implementation about 10x faster than autograd.
- code: not checked
- fit_to_us: reuse_idea - effort L - portable lesson: put the labelled hypothesis into the normaliser so that the gold path never leaves the beam during training (gold-in-beam).
- compat_needs: additive path scores, logadd merging of equal states.
- compat_conflicts: continuous positions need a clustering rule to define "equal states" (our DAG conversion).
- tried_before: none
- verified: checked

### CARD MH43 - Differentiable particle filters: gradient stop at resampling (Jonschkowski) and soft resampling (Karkus)
- papers: Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors (Jonschkowski, Rastogi, Brock, RSS 2018) https://arxiv.org/abs/1805.11122; Particle Filter Networks with Application to Visual Localization (Karkus, Hsu, Lee, CoRL 2018) https://arxiv.org/abs/1805.08975 (both read via ar5iv)
- depth: M
- stage: S4, S8
- idea: Resampling is the beam cut of a particle filter. Jonschkowski: resampling by stochastic universal sampling is not differentiable and simply stops the gradient after one loop iteration, so effects of earlier steps on the current belief are neglected. Karkus: soft resampling draws ancestors from a mixture of the weight distribution and a uniform one and corrects with importance weights, keeping a gradient path.
- formulation: Jonschkowski loss = -log of the Gaussian-mixture density of the true state under the weighted particles; components pre-trained, then end to end on overlapping subsequences (length 20 global localisation, 50 KITTI); 100 particles train, 1000 test. Karkus: q(k) = alpha * w_k + (1 - alpha)/K, new weight w' = w / (alpha w + (1 - alpha)/K) (alpha value not extracted); K = 30 particles in training with no resampling there; squared-error loss on the weighted mean pose; BPTT limited to 4 steps (1 / 2 / 4 steps: 73 / 79 / 79 % success).
- input / output: weighted particle set.
- training: end to end through the filter.
- differentiable: partial - ancestor choice has no gradient (Jonschkowski); soft resampling keeps the weight dependency but stays biased.
- sensors_data: simulated mazes, KITTI visual odometry (Jonschkowski); House3D (Karkus).
- results: KITTI translational error (m/m), test 100: BKF 0.2062, DPF independent 0.1901, end to end 0.1467 +- 0.0149; House3D tracking RMSE PF-net 40.5 cm vs histogram filter 92.0, LSTM 66.9, odometry 109.4. Resampling helped under high uncertainty (39 -> 42 %) but hurt tracking in training (79 -> 75 %). Neither paper discusses particle collapse or bias explicitly.
- compute: about 0.6 ms per particle per step (RGB PF-net).
- code: not checked
- fit_to_us: reuse_idea - effort S - cheap but biased; at least shows that training with and without the cut differ by a few percent in a long-horizon task.
- compat_needs: weights over a particle set.
- compat_conflicts: bias; in the Karkus experiments resampling during training hurt tracking success (79 -> 75 %) while helping under high uncertainty (39 -> 42 %).
- tried_before: none
- verified: checked

### CARD MH44 - Differentiable particle filter via entropic optimal transport (resampling with gradient)
- papers: Differentiable Particle Filtering via Entropy-Regularized Optimal Transport (Corenflos, Thornton, Deligiannidis, Doucet, ICML 2021) https://arxiv.org/abs/2102.07850 (read via ar5iv; last 45 k characters not read)
- depth: M
- stage: S4, S8
- idea: Replace the ancestor draw by the entropic optimal-transport map from the weighted particle set to the uniform one: each new particle is a weighted average of the old ones, so gradients reach all particles.
- formulation: X~_i = N * sum_k P^OT_{eps,i,k} X_k (barycentric projection, Eq. 13); stabilised Sinkhorn (Feydy et al.), cost rescaled so eps is scale-free; eps = 0.5 default; O(N^2) per iteration; gradients by gradient stitching on the converged potentials or autodiff through the loop. Ignoring the resampling gradient gives a biased ELBO gradient (Prop. 4.1); the OT filter is consistent for eps_N = o(1/log N) under compactness assumptions (Prop. 4.3); the likelihood estimate is biased (only affine functions preserved). Soft resampling (alpha = 0.1) performed worse than plain PF and OT filter.
- input / output: weighted particle set -> equally weighted set.
- training: end to end (ELBO, proposal learning, VRNN, localisation).
- differentiable: yes (bias controlled by eps).
- sensors_data: linear Gaussian SSM, polyphonic music, maze localisation.
- results: proposal learning (d = 25): parameter RMSE 0.11 for DPF (N = 25) vs 0.22 for PF (N = 500), ESS about 60 % vs 25 %; JSB test ELBO DPF -7.67, PF -7.92, soft PF -8.17; maze test MSE DPF 3.55 / 4.65 / 4.44 vs PF 10.71 / 11.86 / 12.88 and soft PF 9.14 / 10.12 / 11.42.
- compute: O(N^2) per Sinkhorn iteration in training; plain O(N) filter afterwards.
- code: not checked
- fit_to_us: reuse_idea - effort M - the beam cut with gradient to all nodes; barycentric merging averages positions, which blends different lanes unless the cost makes the transport local.
- compat_needs: weights and a cost in a vector space.
- compat_conflicts: averaged particles of different lanes are off-lane; bias; O(N^2).
- tried_before: none
- verified: checked

### CARD MH45 - Differentiable particle filtering without modifying the forward pass (stop-gradient reweighting)
- papers: Differentiable Particle Filtering without Modifying the Forward Pass (Scibior, Wood, 2021) https://arxiv.org/abs/2106.10314 (read via ar5iv; last 28 k characters not read)
- depth: M
- stage: S4, S8
- idea: Keep the hard resampling of the forward pass. Stop the gradient of the weights flowing into the resampling distribution and multiply each resampled particle's weight by v/stopgrad(v), which equals 1 in the forward pass; gradients then flow into the resulting weights.
- formulation: automatic differentiation of the log-marginal-likelihood estimate recovers the score-function estimator of Poyiadjis et al.; unbiased for expectations under the unnormalised posterior (Theorem 1), consistent for the normalised one; the common practice of dropping resampling terms has a bias that does not vanish (citing Corenflos); marginal variant reduces variance from quadratic to linear in sequence length at quadratic cost in the particle number.
- input / output: weighted particles.
- training: linear Gaussian SSM (200 steps), VRNN on occluded 2-D images (40 steps).
- differentiable: partial - forward hard, gradient to weights through the stop-gradient trick.
- sensors_data: synthetic and image sequences.
- results: improves over standard PF with few particles and converges faster with many; on VRNN standard PF fits training data better but overfits, the new estimator is better on test; no numbers in the excerpt.
- compute: small extra cost; the forward pass is unchanged.
- code: not checked
- fit_to_us: reuse_idea - effort S - pattern for "hard top-M forward, multiplicative node weights carry the gradient of the downstream loss to the survivors' scores"; inconclusive for learning the proposal.
- compat_needs: multiplicative weights on survivors.
- compat_conflicts: dropped candidates still receive no gradient.
- tried_before: none
- verified: checked

### CARD MH46 - Straight-through estimators, Gumbel-softmax, ReinMax and the bias of the surrogate gradient
- papers: Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation (Bengio, Leonard, Courville, 2013) https://arxiv.org/abs/1308.3432 (not opened); Categorical Reparameterization with Gumbel-Softmax (Jang, Gu, Poole, ICLR 2017) https://arxiv.org/abs/1611.01144; Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets (Yin et al., ICLR 2019) https://arxiv.org/abs/1903.05662; Bridging Discrete and Backpropagation: Straight-Through and Beyond (Liu et al., NeurIPS 2023, ReinMax) https://arxiv.org/abs/2304.08612 (last three read via ar5iv)
- depth: M (three), A (Bengio)
- stage: S4, T
- idea: A hard decision in the forward pass with a surrogate derivative in the backward pass. Gumbel-softmax: y_i = exp((log pi_i + g_i)/tau) / sum_j exp((log pi_j + g_j)/tau); the ST variant uses the one-hot sample forward and the soft sample backward. Yin et al.: coarse gradients with a vanilla or clipped ReLU surrogate are positively correlated with the true population gradient (descent direction), the identity surrogate is not. ReinMax: ST is a first-order (forward Euler) approximation of the gradient w.r.t. the logits; the Heun-type correction is second-order.
- formulation: ST gradient = (df/dD) * d softmax/d theta. ReinMax: grad = 2 * grad^((pi+D)/2) - 0.5 * grad^ST, via pi_0 = softmax(theta), D sample, pi_1 = (D + softmax_tau(theta))/2 with stop-gradient, pi_2 = 2 pi_1 - 0.5 pi_0 then the ST trick. Gumbel-softmax schedule tau = max(0.5, exp(-r t)) every N in {500, 1000} steps, r in {1e-5, 1e-4}; small tau gives near one-hot samples with high gradient variance, large tau smooth but biased. ST and ReinMax work better with tau >= 1; ST Gumbel-softmax with tau <= 1.
- input / output: logits -> hard one-hot / k-hot.
- training: end to end; ReinMax costs about the same time and memory as ST (5.2 s per epoch on MNIST-VAE).
- differentiable: partial - the discrete choice has no gradient; the surrogate is biased (ST first-order, ST Gumbel-softmax biased).
- sensors_data: SBN / VAE on MNIST, ListOps, polynomial programming, CIFAR ResNet-20 quantisation.
- results: test NLL SBN (Cat.): ST 61.8, Gumbel-softmax 59.0, ST Gumbel-softmax 59.7; VAE (Cat.): ST 110.9, Gumbel-softmax 101.5, ST Gumbel-softmax 107.8. ResNet-20 4-bit activations: identity surrogate 54.16 %, vanilla ReLU 86.59 %, clipped ReLU 91.24 % (2-bit: 46.52 / 48.05 / 88.39). ListOps accuracy ReinMax 68.07 +- 1.18, STGS 67.30 +- 2.50, ST 66.26 +- 0.48; MNIST-VAE -ELBO ReinMax 103.21, ST 116.72. Yin et al.: a poor surrogate "generates coarse gradients incompatible with the energy landscape" (instability); proofs hold for the population loss (infinite samples) only.
- compute: Gumbel-softmax / ST / ReinMax about equal cost.
- code: not checked
- fit_to_us: reuse_idea - effort S - the generic fallback for any hard keep / pick decision in the tree; the choice of surrogate matters (identity can fail).
- compat_needs: a differentiable surrogate of the decision.
- compat_conflicts: bias; sensitivity to tau and surrogate shape.
- tried_before: none
- verified: checked (Bengio 2013: from memory)

### CARD MH47 - Sparse top-k / top-1 routing with load balancing (Switch Transformer; noisy top-k gating)
- papers: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus, Zoph, Shazeer, JMLR 2022) https://arxiv.org/abs/2101.03961 (read via ar5iv; last 28 k characters not read); Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017) https://arxiv.org/abs/1701.06538 (not opened; described in the Switch paper)
- depth: M (Switch), not read (Shazeer)
- stage: S4, T
- idea: A router picks the highest-probability expert per token; the layer output is p_i(x) * E_i(x), so the gate value keeps the router differentiable although the choice is hard; an auxiliary loss keeps the load balanced and avoids collapse onto few experts.
- formulation: p_i(x) = softmax(W_r x); output p_i(x) E_i(x) for the argmax expert; loss_aux = alpha * N * sum_i f_i * P_i, f_i = fraction of tokens dispatched to i, P_i = mean router probability, alpha = 1e-2; capacity = tokens/N * factor (1.0 - 1.25), overflow tokens pass by the residual (<1 % dropped). Shazeer: noisy top-k (k >= 2) with separate importance and load losses (from the Switch description). Instabilities: hard routing and bfloat16 in the router softmax; remedies: float32 router, initialisation scaled down 10x, expert dropout 0.4 in fine-tuning; Switch-XXL still sometimes unstable.
- input / output: token features -> expert assignment.
- training: end to end; regime `none`.
- differentiable: partial - the choice is constant, gradient through the gate value of the chosen expert.
- sensors_data: C4 text, multilingual (mT5 comparison).
- results: over 7x pre-training speed-up vs T5-Base at equal FLOPs per token; the page summary quotes steps 60 k and 450 k for a 64-expert Switch-Base against T5-Base, "about 7.5x" in steps (the direction of that sentence is ambiguous in the summary); dropped tokens typically < 1 %.
- compute: sparse; routing cost small.
- code: not checked
- fit_to_us: reuse_idea - effort S - the load-balancing term is a documented antidote to children / experts that never get used (slots collapsing), usable as "slot usage" regulariser over a batch of tips.
- compat_needs: batch-level statistics f_i, P_i over the b slots.
- compat_conflicts: forces uniform slot use, whereas forks are rare among tips - a uniform-usage prior is wrong for the second slot; use only as a weak term (the rate of forks per tip in our data was not looked up here).
- tried_before: none
- verified: checked

### CARD MH48 - Differentiable token pruning by a learned keep mask and attention masking (DynamicViT)
- papers: DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification (Rao et al., NeurIPS 2021) https://arxiv.org/abs/2106.02034 (abstract and search summaries only; ar5iv conversion failed); extension: Dynamic Spatial Sparsification for Efficient Vision Transformers and Convolutional Neural Networks https://arxiv.org/abs/2207.01580 (search summary only)
- depth: A
- stage: S4, T
- idea: A lightweight prediction module scores token importance at several layers and prunes hierarchically; training stays end to end through an attention masking strategy that blocks interactions of pruned tokens differentiably (shapes stay fixed in training; tokens are removed physically at inference).
- formulation: not read. From the search summaries: losses cross-entropy + distillation toward the unpruned backbone; extension: ratio loss weight 2, KL and distillation weights 0.5, keep ratios geometric [rho, rho^2, rho^3]. A Gumbel-softmax sampled binary mask (from memory, not confirmed on the pages seen).
- input / output: token sequence -> pruned token sequence.
- training: cross-entropy + distillation + ratio loss; regime `none`.
- differentiable: partial (binary mask with a differentiable surrogate; from memory).
- sensors_data: ImageNet.
- results: about 31 - 37 % FLOPs reduction and over 40 % throughput gain with an accuracy drop within 0.5 % (poster page summary).
- compute: as above.
- code: https://github.com/raoyongming/DynamicViT (named in a search summary)
- fit_to_us: reuse_idea - effort M - the masking trick is the standard way to prune with fixed tensor shapes in training: dropped tree nodes stay in the tensor but are masked out of attention among siblings and from the loss.
- compat_needs: attention among nodes (our cascade stage C).
- compat_conflicts: needs a target keep ratio; details unread.
- tried_before: none
- verified: partly

### CARD MH49 - Beam-Search Optimization (BSO): margin loss on the beam, reset when the gold prefix falls off
- papers: Sequence-to-Sequence Learning as Beam-Search Optimization (Wiseman, Rush, EMNLP 2016) https://arxiv.org/abs/1606.02960 (read via ar5iv)
- depth: M
- stage: T, L
- idea: Train a sequence model with the beam search it is decoded with: whenever the gold prefix is not within the beam (margin violation), a margin loss pushes the gold prefix above the K-th beam entry; the beam is then reset to contain the gold prefix (LaSO-style) and search continues.
- formulation: L(f) = sum_t Delta(y_hat_{1:t}^(K)) * [ 1 - f(y_t, h_{t-1}) + f(y_hat_t^(K), h_hat_{t-1}^(K)) ]; Delta = 0/1 (word ordering, parsing) or 1 - smoothed sentence BLEU (MT); f is an unnormalised score (removes label bias); pre-training with word-level cross-entropy needed (random init failed); beam curriculum from 2, +1 every 2 epochs up to K_tr; constrained beam search (ConBSO) with hard constraints in the successor function; gradients flow through the gold and violating prefixes (not through the whole beam); Adagrad, batch 64.
- input / output: sequence model with scores; beam of prefixes.
- training: regime `free_rollout` with reset (gold-in-beam).
- differentiable: partial - top-K selection constant; gradient through the scores of gold and violating prefixes.
- sensors_data: word ordering, dependency parsing, German-English MT.
- results: word ordering BLEU (K_te = 10): seq2seq 31.0, BSO 34.3, ConBSO 34.5; dependency parsing UAS / LAS: 88.66 / 84.33 -> 91.17 / 87.41 (BSO), 91.57 / 87.26 (ConBSO); MT BLEU (K_te = 5): 24.03 -> 26.36.
- compute: tokens/s 13,038 (seq2seq) vs 1,985 (K_tr = 2) and 1,458 (K_tr = 6); the stated 3.3x overhead is inconsistent with the K_tr = 2 figure (about 6.6x).
- code: not checked
- fit_to_us: reuse_idea - effort M - "reset" = insert the GT-nearest node into the beam when it was cut; margin loss needs an unnormalised score (our validity).
- compat_needs: scores comparable across beam entries; a gold prefix (nearest-node sequence).
- compat_conflicts: the gold prefix must exist as a node sequence in the tree.
- tried_before: none
- verified: checked

### CARD MH50 - Learning beam search policies via imitation learning (beam-aware losses, continue / reset / stop)
- papers: Learning Beam Search Policies via Imitation Learning (Negrinho, Gormley, Gordon, NeurIPS 2018) https://arxiv.org/abs/1811.00512 (read via ar5iv; experiments section not reached)
- depth: M (partial)
- stage: T, L
- idea: Treat the beam as part of the model: a scoring function ranks the neighbours of the beam elements, an oracle cost-to-go defines which neighbour is best, and surrogate losses on the beam are minimised on trajectories collected by the learned policy.
- formulation: beam = up to k nodes; candidates A_b = all neighbours; keep top-k by score s(v, theta); cost-to-go c*(v); beam cost = min over elements; cost increase when the best completion falls out of the beam. Losses: perceptron (first) = early update; perceptron / margin (last); cost-sensitive margin (last) close to BSO; upper bound on expected transition cost; log losses normalised over the beam or over all neighbours. Data collection: oracle, stop (first cost increase), reset (to the oracle's best node, as LaSO), continue (keep following the learned policy, as DAgger, the new strategy), mixtures with weight beta. Theory: no-regret reduction (DAgger-style), bounds for stop and reset depend on how often the roll-in deviates.
- input / output: any search space with scores and an oracle.
- training: regime `dagger` (continue) / `free_rollout`; Adam or SGD; losses differentiable w.r.t. the scores (how gradients pass the beam selection is not detailed).
- differentiable: partial - selection constant, loss on scores.
- sensors_data: NLP tasks (not reached in the text read).
- results: none seen.
- compute: not stated.
- code: not checked
- fit_to_us: reuse_idea - effort M - the clearest statement of training with the beam in the loop; continue (own beam) is our free roll-out, reset is gold-in-beam; the oracle cost-to-go can be a point-to-polyline residual of the subtree.
- compat_needs: an oracle cost per node (distance to the GT lane graph).
- compat_conflicts: needs the node scores to be comparable; loss choices matter (not tested for geometry).
- tried_before: none
- verified: partly

### CARD MH51 - Scheduled sampling (mix ground-truth and own predictions with a decaying probability)
- papers: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks (Bengio, Vinyals, Jaitly, Shazeer, NeurIPS 2015) https://arxiv.org/abs/1506.03099 (read via ar5iv)
- depth: M
- stage: T
- idea: Per token flip a coin: with probability eps_i feed the ground-truth previous token, else the model's own sample or argmax; eps_i decays over training (curriculum from easy to realistic).
- formulation: schedules linear eps_i = max(eps, k - c i), exponential eps_i = k^i, inverse sigmoid eps_i = k/(k + exp(i/k)); the coin is per token (per sequence performed much worse); gradients are NOT propagated through the sampled token (left to future work); the term "exposure bias" is not used, the paper speaks of errors amplified early in the sequence.
- input / output: RNN decoders.
- training: regime `teacher_forcing` -> `free_rollout` curriculum.
- differentiable: partial - no gradient through the sampled token.
- sensors_data: MSCOCO captioning, WSJ constituency parsing, speech frame prediction.
- results: captioning (BLEU-4 / METEOR / CIDEr): baseline 28.8 / 24.2 / 89.5, scheduled sampling 30.6 / 24.3 / 92.1, always sampling 11.2 / 15.7 / 49.7; parsing F1: baseline 86.54, scheduled sampling 88.08, always sampling invalid trees; speech: always sampling 34.6 / 35.8 frame error rate vs baseline 15.0 (next-step) / 46.0 (decoding).
- compute: no overhead beyond sampling.
- code: not checked
- fit_to_us: reuse_idea - effort S - ramp the share of own children (instead of GT-nearest children) fed to the next level from 0 to 1; pure free roll-out from the start is documented as poor in captioning and parsing.
- compat_needs: a decoder that accepts either GT or own input at each level.
- compat_conflicts: the sampled input carries no gradient.
- tried_before: none
- verified: checked

### CARD MH52 - Polygon-RNN++ (beam of candidate polygons ranked by a separately trained evaluator; RL against exposure bias)
- papers: Efficient Interactive Annotation of Segmentation Datasets with Polygon-RNN++ (Acuna, Ling, Kar, Fidler, CVPR 2018) https://arxiv.org/abs/1803.09693 (read via ar5iv)
- depth: M
- stage: S4, T
- idea: A ConvLSTM predicts polygon vertices one by one as a classification over a 28 x 28 grid (plus an end token); cross-entropy pre-training, then self-critical RL on IoU so the decoder trains on its own samples; at inference K = 5 first-vertex candidates each run a beam, and an evaluator network (trained with MSE to predict the IoU of a polygon) picks the best.
- formulation: teacher-forced training with a distance-transform-smoothed target (Manhattan distance truncated at 2); RL: reward = IoU of predicted and GT masks, baseline = reward of the greedy polygon, temperature 0.6; evaluator candidates sampled at temperature 0.3 for diversity, trained separately after RL converged; beam width B not stated for the main results (fast variant B = K = 1, 295 ms per object).
- input / output: image crop -> polygon vertices.
- training: regime `teacher_forcing` then `rl`; selection by a learned verifier.
- differentiable: no for the RL path (reward); evaluator is a separate regressor.
- sensors_data: Cityscapes (GT boxes, automatic mode).
- results: cumulative ablation mean IoU: residual Polygon-RNN 62.16, + attention 65.43, + RL 67.17, + evaluator 70.21, + GGNN 71.38; evaluator with greedy sequences 69.72 vs beam search without evaluator 67.49.
- compute: 295 ms per object for B = K = 1.
- code: not checked
- fit_to_us: reuse_idea - effort M - a learned verifier (IoU predictor) to choose among beam candidates and RL against exposure bias are both used; RL is outside our "no argmax on the training path" rule.
- compat_needs: IoU-like reward; evaluator network.
- compat_conflicts: RL / sampling; two-stage training.
- tried_before: none
- verified: checked

### CARD MH53 - On-policy closed-loop training for multimodal prediction (execute one mode, detach across transitions)
- papers: Rectify, Don't Regret: On-Policy Closed-Loop Training for Multimodal Trajectory Prediction (Yadav, Bohn, Meisen, arXiv 2603.23393, v2 Sept 2026) https://arxiv.org/abs/2603.23393 (arXiv HTML read; code https://github.com/harshy105/CL-Traj). A companion paper (arXiv 2603.24155) is marked withdrawn and is not used.
- depth: M
- stage: T, L
- idea: A multimodal predictor is rolled out in closed loop during training: each predicted mode is supervised only on states its own execution produced, and gradients are cut at the executed transitions so the model learns to recover from drift instead of "regretting" an earlier prediction.
- formulation: receding horizon: each predicted mode is executed for T_sim (default 2.0 s, T_pred = 6 s), the model re-predicts from the resulting state; stop-gradient on the executed ego positions. The open-loop sample (n = 0) supervises classification (mode probabilities) and regression of all M modes; closed-loop samples (n >= 1) supervise only regression of the executed mode m*, with the original logged future as target; m* is chosen from the open-loop sample by L2 distance to the ground truth (WTA) and fixed for the rollout, so training unrolls ONE mode (cost scales with replanning count, not M x replanning). At test time all M modes are rolled out. An off-policy variant that re-selects the best mode each step was worse, especially at T_sim = 0.5 s. A fully differentiable closed loop ties the policy to the training replanning rate (shown in a figure only).
- input / output: LMFormer-D (decoder-only) and an unmodified QCNet.
- training: regime `free_rollout` (on-policy, detached) mixed with open-loop teacher signal; 5 seeds main, 3 ablations.
- differentiable: partial - detached by design.
- sensors_data: nuScenes, Argoverse 2, DeepScenario.
- results: closed-loop evaluation at T_sim = 0.5 s, collision % open-loop -> closed-loop training: nuScenes 3.95 -> 3.08, DeepScenario 4.46 -> 2.92, Argoverse 2 (QCNet) 2.84 -> 2.35; off-road % 5.65 -> 1.86, 22.28 -> 8.62, 16.67 -> 15.07; at T_sim = 6.0 s the two are comparable; abstract: up to 35 % fewer collisions, 67 % fewer off-road.
- compute: closed-loop samples add cost proportional to replanning frequency; LMFormer-D has 45 % fewer parameters and 62 % fewer FLOPs than LMFormer.
- code: https://github.com/harshy105/CL-Traj
- fit_to_us: reuse_idea - effort M - the only direct evidence for free roll-out with multiple hypotheses: execute the GT-nearest hypothesis (WTA from the teacher sample), supervise regression only for it, keep the classification (score) signal from the teacher-forced sample, detach across levels.
- compat_needs: a one-mode rollout path through the tree (the GT-nearest child continues), a separate teacher-forced pass for scores.
- compat_conflicts: detaching cuts the path from the downstream loss to the early children (our E121 finding that planner gradients through positions give no gain fits).
- tried_before: none
- verified: checked

### CARD MH54 - Differentiable beam search for graph generation (decoding-aware training for information extraction)
- papers: Information Extraction with Differentiable Beam Search on Graph RNNs (El Khbir, Tomeh, Charnois, LREC-COLING 2024) https://aclanthology.org/2024.lrec-main.796/ (abstract only)
- depth: A
- stage: T
- idea: Graph generation cast as autoregressive sequence labelling; training is made decoding-aware with a differentiable version of beam search to reduce exposure bias; a new way to quantify exposure bias is proposed.
- formulation: not read (the page does not say which relaxation or loss).
- input / output: not read.
- training: decoding-aware; regime `free_rollout`.
- differentiable: yes (abstract).
- sensors_data: ACE05, CoNLL04.
- results: gains over a version without decoding-aware training (abstract); numbers not seen.
- compute: not stated.
- code: none found
- fit_to_us: reuse_idea - effort M - recent evidence that differentiable-beam training is used for graph-structured outputs, but only in text.
- compat_needs: not read.
- compat_conflicts: not read.
- tried_before: none
- verified: partly

### CARD MH55 - Trexplorer Super (recurrent DETR tracking of tree centrelines; super-trajectory training; duplicate branches)
- papers: Trexplorer Super (Naeem et al., MICCAI 2025, arXiv 2507.10881; full title not on the page read) https://arxiv.org/abs/2507.10881 (arXiv HTML read); the original Trexplorer: Recurrent DETR for Topologically Correct Tree Centerline Tracking (MICCAI 2024; PDF not decodable); RefTr: Recurrent Refinement of Confluent Trajectories for 3D Vascular Tree Centerlines (Naeem et al., arXiv 2511.20823) https://arxiv.org/abs/2511.20823 (abstract only)
- depth: M (Trexplorer Super), A (RefTr)
- stage: S4, S6, T
- idea: A DETR decoder tracks each branch breadth-first, predicting the child nodes of the next graph level (end / intermediate / bifurcation); patches are placed at the endpoints of the graph built so far and the query outputs of the previous patch are carried over. Known problems of the first version: duplicate branches and premature termination.
- formulation: Hungarian loss between predicted and GT nodes with self-attention among queries to suppress duplicates (individual loss terms not specified); history = past-trajectory token of up to 10 previous node positions; Super Trajectory Training: a sample is a super trajectory of 54 nodes split into six sub-trajectories of 10 nodes with shared link nodes, query outputs passed between them (so the model trains on its own carried state, not only GT); target augmentation samples Laplace offsets at bifurcations (scale proportional to radius) to create more bifurcation targets; Focal Cross Attention (high-resolution features, attention restricted to a focal region around the tracked branch); inference: 26 queries assigned at a bifurcation, some discarded, at most 196 tokens tracked concurrently, no NMS, no post-processing. The page does not describe teacher forcing or multi-hypothesis methods.
- input / output: 3-D image patches -> tree nodes.
- training: regime `hungarian_set` per step; STT gives a form of free rollout inside super trajectories.
- differentiable: partial - Hungarian assignment; carried queries are differentiable.
- sensors_data: synthetic (368 / 32 / 100), ATM'22 (220 / 16 / 60), Parse 2022 (72 / 8 / 20) CT volumes.
- results: point F1 (Vesselformer / Trexplorer / Trexplorer Super): synthetic 48.18 / 39.40 / 77.83; ATM'22 26.77 / 3.34 / 60.45; Parse 2022 16.43 / 10.01 / 39.46; branch F1 15.95 / 26.26 / 77.12 (synthetic), 1.95 / 3.34 / 41.15 (ATM'22); Betti errors zero for both Trexplorer variants.
- compute: not reported.
- code: https://github.com/RomStriker/Trexplorer-Super (paper CC BY-NC-ND 4.0; code licence not stated on the page)
- fit_to_us: baseline - effort L - closest published tree tracker with several children per step; shows duplicates and early stops as the failure modes of the matched version, and training on carried (own) state as the fix.
- compat_needs: Hungarian; 3-D patches.
- compat_conflicts: Hungarian; patch-based inputs.
- tried_before: E94-E99 (learned trace network, teacher forced)
- verified: checked

### CARD MH56 - Modified multiple hypothesis tracking for tree extraction (ranked local hypotheses, search-depth tree)
- papers: Extracting Tree-structures in CT data by Tracking Multiple Statistically Ranked Hypotheses (Selvan et al.) https://arxiv.org/abs/1806.08981 (read via ar5iv)
- depth: M
- stage: S4, S5
- idea: Classical hypothesis-tree tracking: at regularly spaced steps tubular templates are fitted in a few directions, local hypotheses are ranked, and a tree of depth d of hypotheses is built; decisions are deferred until a branch scores well across several steps.
- formulation: step length = scaled local radius; search angle 70 degrees with 2 angles for airways; template profile steepness 8, weighted least squares; score = 1 / rank among the N hypotheses at that step (best = 1, removes scale dependence); global hypothesis = d local hypotheses scored by the average rank score; bifurcations: spectral clustering of hypotheses; tree state saved and tracking resumed per new branch; global threshold 0.7 (airways); search depth 6 (airways), 4 (coronary); after a decision, hypotheses that are not children of the best node are discarded.
- input / output: CT volume intensities (or probability images) -> centreline tree.
- training: none (classical); parameter tuning at depth 6 took about 88 hours.
- differentiable: no.
- sensors_data: airway CT (16 test scans), coronary CT.
- results: mean centreline error 1.93 mm vs 3.96 mm original MHT, 3.37 mm region growing on intensity, 2.06 mm region growing on probability images (p = 0.1 vs the last); coronary overlap 91.8 % vs 67.6 % original MHT; about 15 s per airway scan, 3 s per vessel.
- compute: depth 4 - 6 feasible, trade-off between under-segmentation and run time.
- code: not checked
- fit_to_us: baseline - effort M - the non-learned counterpart of our tree: local template sampling, rank-based scores, deferred decision by depth; rank scores 1/R are a calibration-free alternative to raw validity.
- compat_needs: local scores per hypothesis.
- compat_conflicts: not trainable.
- tried_before: none
- verified: checked

### CARD MH57 - DTPP / TPP (pre-generated ego trajectory tree, conditional prediction, learned cost, joint differentiable training)
- papers: DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving (Huang, Karkus, Ivanovic, Chen, Pavone, Lv, ICRA 2024) https://arxiv.org/abs/2310.05885 (arXiv HTML read); Tree-structured Policy Planning with Learned Behavior Models (Chen, Karkus, Ivanovic, Weng, Pavone, 2023) https://arxiv.org/abs/2301.11902 (abstract level)
- depth: M (DTPP), A (TPP)
- stage: S4, S10, L
- idea: The ego tree of candidate trajectories is PRE-GENERATED (lane-centric target sampling), a conditional prediction model decodes all branches in one pass, a learned cost scores each branch, and prediction and cost are trained jointly; pruning to the top nodes halves the run time.
- formulation: tree: two stages (0 - 3 s, 3 - 8 s); stage 1 from three reference paths with up to 30 sampled target states (third-order polynomial speed profiles), keep the top 5 nodes, expand each with 6 targets for stage 2; infeasible trajectories dropped, random discarding above the node limit M = 30. Cost per branch = horizon-averaged weighted sum of ego features, learned interaction features and a Gaussian RBF collision feature, weights from an MLP on the ego encoding (10 weights); P(s_m) = softmax(-c). Loss: smooth L1 for the prediction on the branch closest to the GT ego trajectory + cross-entropy of softmax-normalised plan scores against the GT label; weights beta1 = 1.0, beta2 = 0.2, IRL alpha = 0.1; the cost loss back-propagates into the prediction module; gradient through the branch selection not stated.
- input / output: 10 agents, ego, 50 map elements (transformer, D = 256, 3 layers) -> branch trajectories, costs.
- training: nuPlan, RTX 3080, batch 16, 20 epochs; regime `matching_free` (nearest branch as label).
- differentiable: partial - candidate generation and pruning are rule-based; cost and prediction are differentiable.
- sensors_data: nuPlan (closed loop), vector inputs.
- results: NR / R score: DTPP 0.8964 / 0.8978 (collision 0.025), TPP 0.7388 / 0.7699, PDM 0.9061 / 0.9150, IDM 0.6396 / 0.6168; tree vs single stage: single-3 s 0.7087 / 0.7044, single-8 s 0.7799 / 0.7558; joint vs separate training (no pruning) 0.8948 / 0.8933 vs 0.8134 / 0.8106.
- compute: 98.0 ms with pruning vs 199.6 ms without.
- code: not checked
- fit_to_us: reuse_idea - effort M - the nearest published "pre-generated hypothesis tree + learned scoring + nearest-branch label + joint differentiable training"; pruning the top nodes per stage with a rule is accepted practice.
- compat_needs: a candidate generator; a nearest-branch label; a node-wise cost.
- compat_conflicts: the tree is not learned (rule-based), nodes are ego trajectories, not map elements.
- tried_before: none
- verified: checked

### CARD MH58 - Multiple hypothesis tracking with learned parts (online appearance models; neural gating of the hypothesis tree)
- papers: Multiple Hypothesis Tracking Revisited (Kim, Li, Ciptadi, Rehg, ICCV 2015) https://openaccess.thecvf.com/content_iccv_2015/html/Kim_Multiple_Hypothesis_Tracking_ICCV_2015_paper.html (PDF returned 403, listing and search summary only); Multi-object Tracking with Neural Gating Using Bilinear LSTM (Kim, Li, Rehg, ECCV 2018) https://openaccess.thecvf.com/content_ECCV_2018/html/Chanho_Kim_Multi-object_Tracking_with_ECCV_2018_paper.html (abstract and survey summary only)
- depth: A
- stage: S4, S8
- idea: Track-oriented MHT keeps a tree of association hypotheses per track and prunes it; Kim 2015 adds online-trained regularised-least-squares appearance models per hypothesis branch; Kim 2018 uses a bilinear LSTM whose affinity score gates (prunes) branches of the hypothesis tree and scores tracks by appearance and motion.
- formulation: not read.
- input / output: detections per frame -> tracks.
- training: Kim 2018: LSTM trained on tracklet appearance features; the tree itself is not trained.
- differentiable: no for the tree (gating network yes).
- sensors_data: MOT16 / MOT17 (camera).
- results: state-of-the-art near-online performance on MOT16 / MOT17 (abstract); numbers not seen. Searches for a learned or differentiable MHT with trainable hypothesis tree found none; transformer trackers (TrackFormer, MOTR, TADN) are not hypothesis trees.
- compute: not stated.
- code: none found
- fit_to_us: reuse_idea - effort M - confirms the pattern "learned gate decides which branch of a hypothesis tree survives" without end-to-end training of the tree.
- compat_needs: not read.
- compat_conflicts: not read.
- tried_before: none
- verified: partly

## 3. Cluster compatibility notes

Marking: statements without a tag come from the cards; "[own inference]" marks my reasoning about our pipeline that no paper read here states.

### 3.0 Key findings (short)

1. The published matching-free ways to train b hypotheses against a variable number of GT elements are nearest-hypothesis (WTA / Voronoi) assignment with an independent score head (rMCL, aMCL: MH01, MH02) and nearest-anchor / nearest-slot assignment with CE or BCE (MH10, MH12, MH16). Starvation of the idle hypotheses when GT is scarcer than hypotheses is the documented failure; the fixes (cooling, catch-up term, eps floor, k schedule, fixed slots, many modes + aggregation: MH02 - MH05, MH10 - MH13) were shown with at least 3 hypotheses and none for b = 2 with mostly single-GT inputs.
2. The scores in this literature mean "responsibility / cell mass", not "exists on a lane". Existence-and-quality targets (MH29, MH19) and a best-per-GT duplicate label (MH31) are the matching-free ingredients for a validity score and for redundancy suppression.
3. Hard top-M with a local label for every candidate needs no gradient to dropped nodes. Every differentiable-pruning construction (soft beam, SOFT top-k, OT resampling, entmax) was shown on discrete or latent states; those that average states (MH41, MH44) conflict with geometry. The transferable parts are weight-carrying survivors (MH45, MH47) and gold-in-beam (MH42, MH49, MH50).
4. No precedent trains a multi-hypothesis model free-running from step 1: sequence precedents warm up with teacher forcing or ramp (MH41, MH49, MH51, MH17); the only multi-hypothesis result (MH53) executes one WTA mode per rollout and detaches.
5. No paper trains a geometric b-ary tree with a beam end to end; the closest are DTPP (rule-generated tree, learned cost; MH57), Trexplorer Super (Hungarian, MH55) and PGP (sampled edge policy, MH17).

### 3.1 Interface contracts (what flows in and out)

- Hypothesis-set contract. In every card the exchanged unit is a set H(x) = {(h_k, s_k)}, k = 1..K: h_k a point or trajectory in a metric frame, s_k a score, all produced from one context x. Sources of the set: K heads on a trunk (MH01 - MH07), K learned slot queries (MH11, MH13, MH14), K fixed anchors / intention points (MH10, MH12, MH16, MH18), a dense candidate pool with a heat map (MH15, MH19 - MH21), sampled graph traversals (MH17), a pre-generated tree (MH57). Mapping to ours: x = tip token + bilinear BEV samples; h_k = (position, heading, step class) in ego metres; s_k = validity / keep probability. MH16 samples BEV features at the trajectory coordinates exactly as our `grid_sample` does; no card needs more than that.
- Three kinds of score. (1) Responsibility, normalised ("which hypothesis is the answer"): rMCL after normalisation (MH01), MultiPath / MTR / Wayformer / Villaflor (MH10, MH12, MH13, MH18), entmax (MH35). (2) Existence / quality, independent sigmoid with a target tied to geometric quality: DiffusionDrive (MH16), POTO (MH19), OneNet (MH20), Align-DETR (MH29), Relation Networks (MH31). (3) Path score, additive in log space along a tree: MH41, MH42, MH49, MH50, MH57. Our validity is kind 2, the beam needs kind 3, WTA-trained heads give kind 1. A fork has two true children, so a normalised score needs a two-hot target.
- Label contract. All WTA / anchor / quality methods need only a distance between each hypothesis and each GT element: no ordering, no matching. For a tip the GT elements are the successor polyline(s) (point-to-polyline distance, as in E120) or successor end points per step class. Prediction-aware labels (MH19, MH20, MH21, MH31) additionally read the current scores while labelling, so labels drift during training; MH27 measures this instability for Hungarian labels in DETR.
- Gradient contract: see Table B.

### 3.2 Table A - how GT is assigned to a hypothesis without Hungarian, and how the score is trained

| method (card) | GT -> hypothesis rule | score target | remedy for starved / collapsed hypotheses |
|---|---|---|---|
| MCL / sMCL (MH06) | nearest member per example | none | none (failure named in MH03) |
| relaxed WTA (MH04) | nearest gets 1 - eps, others eps/(M-1), eps = 0.05 | none | eps floor, 1 % dropped predictions |
| EWTA (MH05) | k nearest, k from M to 1 | none | k schedule |
| rMCL (MH01) | each GT to its nearest hypothesis (Voronoi cell), several GT allowed | BCE: hypothesis wins >= 1 GT | scores flag unreliable hypotheses; eps- / top-n- combinations |
| aMCL (MH02) | softmin weights at temperature T, detached | as rMCL | cooling T -> 0 |
| DiverseNet (MH03) | each GT to its nearest, not one-to-one | none | catch-up term on the farthest hypothesis |
| Confident MCL (MH07) | winner by loss | losers: KL to uniform | uniform-KL on losers |
| MultiPath (MH10) | nearest fixed anchor | CE over anchors | fixed anchors |
| MultiPath++ (MH11) | nearest mode by Euclidean distance (learned anchors) | CE of the winner | learned anchors, larger batch, aggregation |
| MTR (MH12) | intention point nearest to the GT endpoint | CE on the positive's probability | static k-means intention queries |
| Wayformer (MH13) | closest mean | CE of the winner | 64 modes, aggregate to 6 |
| QCNeXt / Scene Transformer (MH14) | min displacement, scene level | mixture NLL / CE weight 0.1 | detached proposals, recurrent proposals |
| DenseTNT (MH15) | candidate nearest GT endpoint | BCE on a softmax heat map; set predictor | offline set optimiser as label |
| DiffusionDrive (MH16) | nearest k-means anchor | BCE per anchor | anchors |
| POTO (MH19) | argmax of p^(1-a) IoU^a per GT inside a prior | the quality itself | auxiliary one-to-many, 3D max filter |
| OneNet (MH20) | minimum-cost sample per GT | class cost inside the cost | bipartite matching for crowds |
| YOLOv10 (MH21) | top-1 by a consistent metric | same metric | dual heads |
| Relation Net (MH31) | highest-scoring detection per GT (current scores) | BCE on s0 * s1 | multiple IoU thresholds |
| Align-DETR (MH29) | k candidates, rank-decayed | t = exp(-r/tau) p^a u^(1-a) | - |

### 3.3 Table B - selection operators: what the forward pass does and who receives gradient

| operator (card) | forward | gradient reaches | bias / cost |
|---|---|---|---|
| local per-node loss (detectors, MH19 - MH21, MH31) | no selection needed | every candidate, kept or not | label quality only |
| hard top-k, score multiplies the output (MH47, DETR-style) | hard | scores of kept items via the downstream loss; dropped get none | none for kept; blind to dropped |
| hard beam + sequence loss on gold / violating prefixes (MH42, MH49, MH50) | hard | scores of gold and violating prefixes | needs gold-in-beam (reset, continue, or gold added to the normaliser) |
| soft beam (MH41) | convex combination of states | all candidates | bias vs hard beam, alpha annealed, cost about k x |
| SOFT top-k, soft rank / sort (MH34, MH36) | soft indicator | all candidates | bias O(eps), train / test mismatch |
| perturbed top-k (MH37) | noise-smoothed | all | smoothing bias eps; variance for small M |
| relaxed top-k, SIMPLE (MH38) | relaxed / exact k-subset | all (relaxation / marginals) | biased, low variance |
| entmax / sparsemax (MH35) | sparse weights | support only | exact zeros are dead |
| stop-gradient reweighting (MH45), soft resampling (MH43) | hard resample | weights of survivors | consistent (MH45) / biased (MH43) |
| OT resampling (MH44) | barycentric merge | all | bias eps; blends positions; O(N^2) |
| straight-through, Gumbel-softmax, ReinMax (MH46) | hard sample | all logits via surrogate | first-order bias (ST); second order (ReinMax) |
| Gumbel-top-k + unordered-set estimator (MH39, MH40) | sampled without replacement | all (score function) | unbiased, higher variance |
| attention-mask pruning (MH48, depth A) | masked, fixed shape | gates via surrogate (from memory) | details unread |

### 3.4 Answers to the brief's questions

**(a) b >= 2 children per tip when 1 or 2 GT continuations exist, matching-free.** Patterns in the papers: P1 multi-target WTA with Voronoi cells and a BCE winner score (MH01); P2 DiverseNet loss with catch-up (MH03); P3 slots with a fixed or learned role and a nearest-slot positive (MH10, MH11, MH12, MH16, MH18); P4 a dense pool with a quality argmax per GT plus an auxiliary one-to-many loss (MH19 - MH21, MH25); P5 heat map plus a learned set predictor on pseudo-labels (MH15); P6 a policy over successor edges trained by behaviour cloning on visited edges, forks through the edge visits (MH17). Starvation of the idle hypotheses when GT is scarcer than hypotheses is the documented failure (MH03, MH04, MH05, MH10 toy, MH02); remedies and their evidence are in Table A; all were shown with at least 3 hypotheses on regression problems, none for b = 2 with mostly single-GT inputs.
[own inference] A1 two jobs, no slots: a pool of verified ray endpoints with an independent validity score (target from point-to-polyline closeness, MH29 / MH19 style) and a duplicate label by the Relation-Network rule (among pool points close to the same GT branch the best-scored is positive, the others duplicates, MH31); children = top-b of validity x (1 - duplicate); b is a cap, not a mode count. A2 generated children: rMCL multi-target loss with the nearest-child rule on point-to-branch distance (MH01), plus one of aMCL cooling (MH02), catch-up (MH03) or an eps floor (MH04) in the first epochs; for two GT branches an ordered assignment (children and branches both sorted by bearing, first to first) so that each branch gets its own child - in one dimension the sorted assignment is the optimal one for convex costs (standard optimal-transport fact, not from a paper read here), and it is a sort, not a Hungarian solve; independent nearest assignment (MH01, MH03) can put both branches into the cell of one child. A3 slots from the E94 fan (9 bearings x 3 step classes as anchors, MH10 / MH12 style) with CE / BCE per slot; a straight lane lies on a sector border, so use overlapping sectors or soft targets. A4 guards against the idle second child: cooling or catch-up only early; a hinge repulsion between children (MH08) only at tips with two GT branches and only late (MH08 reports instability when a diversity loss is on from the start); a slot-usage term (MH47) only weakly because forks are rare.

**(b) How hypothesis scores are trained and calibrated.** Trained as: BCE on cell membership (MH01, MH02); CE of the assigned anchor (MH10, MH12, MH13); BCE per anchor (MH16); BCE on a heat map (MH15); quality target p^a * u^(1-a) (MH29, MH19); BCE of "best detection per GT" (MH31). Calibration is rarely a goal: rMCL gives cell masses (Prop. 1 of the paper, point-mass or uniform-in-cell variants); the leaderboard metric Brier-minFDE (MH11, MH13 report it) rewards calibrated mass; Align-DETR says it does not aim at calibration; Selvan's rank score 1/R (MH56) removes the raw scale. Nothing studies the calibration of PRODUCTS of such scores along a tree, which is what a beam ranks by.

**(c) Pruning a beam of M nodes per level with gradient.** Options with who gets gradient: see Table B. [own inference] R1 local supervision of every candidate (as every detector does) plus hard top-M: no gradient to dropped nodes is needed because each node has its own label from the GT graph. R2 gold-in-beam for sequence-level losses (BSO reset, Negrinho continue, Collobert normaliser: MH49, MH50, MH42), gold = the nearest-child sequence. R3 hard forward with multiplicative survivor weights that carry the downstream gradient (MH45, MH47). R4 soft top-M weights (MH34 - MH38) multiplying node contributions, annealed eps, hard at inference. R5 masks in attention among siblings with fixed shapes (MH48). R6 stochastic selection with an unbiased estimator only if an episode-level reward is used (MH39, MH40). Carry weights, do not blend positions: the soft-beam and OT-resampling constructions (MH41, MH44) average states, and an average of two lane points is not on a lane. All demonstrations of differentiable pruning are on discrete or latent states (text, speech, filters); none on geometric nodes.

**(d) Training with free roll-out from the first step (exposure bias) with several hypotheses.** Evidence: scheduled sampling needs a curriculum and "always sampling" is poor in captioning and parsing (MH51); BSO fails from random initialisation and needs cross-entropy pre-training and a beam curriculum (MH49); soft-beam training starts from a cross-entropy model and anneals alpha (MH41); Negrinho et al. name "continue" (follow the learned beam) as the roll-in that fixes the mismatch and give regret bounds for stop / reset (MH50); PGP pre-trains 100 epochs on GT traversals then 100 epochs on policy-sampled paths (MH17); Polygon-RNN++ pre-trains with cross-entropy then self-critical RL (MH52); Trexplorer Super trains on carried query state across sub-trajectories (MH55). The only multi-hypothesis case: Yadav et al. (MH53) execute ONE mode (the one nearest the GT in the teacher sample), supervise only its regression, keep the score signal from the teacher sample and detach across transitions; better than open-loop at short replanning, comparable at 6 s. No paper trains a multi-hypothesis model free-running from step 1; the project plan "free unroll from the first step" has no precedent here and is a risk to test against a warm-up or a ramp (MH51 style).

**(e) Collapse and instability reports.** WTA starvation: MH03 (members receive no training signal), MH04 (dead hypotheses, centroid bias of the fix), MH05 (hypotheses stuck between samples or in the centre; MDN mode collapse and numerical instability), MH10 (toy: min-of-K collapsed to 1 or 2 modes), MH02 (MCL collapsed for some seeds), MH01 (risk, not seen). Overconfident non-winners: MH07. Unstable Hungarian labels: MH27, MH25. Duplicate branches and early stops in tree tracking: MH55. Too many diverse auxiliary heads conflict: MH25 (K = 6 worse than K = 2). Diversity loss destabilises when active from the start: MH08. Hard routing and low precision destabilise the router: MH47. Soft beam with a large constant alpha unstable; a beam that is too small diverges: MH41, MH42. Identity straight-through surrogate can fail: MH46. Learned anchors need richer inputs / bigger batches to beat static ones: MH11; static intention queries beat learnable ones by 4.3 mAP points on a validation subset: MH12.

**(f) Compute cost seen.** K heads on one trunk: negligible (MH01, MH04). Group DETR: about 5 min more per epoch (MH23). H-DETR: +23.1 % training time with R50, no inference cost (MH24). Co-DETR: 70 -> 120 GPU hours with two auxiliary heads (MH25). BSO: about 6.6x slower per token at beam 2 (MH49). Differentiable beam decoder: 35 / 90 / 180 min per epoch at beam 500 / 1000 / 2000 (MH42). Soft beam: about k x cross-entropy (MH41). SOFT top-k: O(n) per Sinkhorn iteration (MH34). OT resampling: O(N^2) per iteration (MH44). DiffusionDrive planning module 7.6 ms (MH16). For M x b <= about 100 candidates every selection operator above is negligible next to BEV sampling.

### 3.5 Known adapters and known conflicts

- Adapter: point-to-polyline distance as the loss inside WTA (E120 already uses it). WTA theory (cells, centroids) is for point losses (MH01, MH04); with a set distance a child may slide along the lane, so step length must come from the verification or a step-class classifier.
- Adapter: normalised <-> independent scores: sigmoid per slot with a multi-hot target (MH16, MH29) or a Fenchel-Young loss with a two-hot target (MH35, from memory for the multi-target form).
- Adapter: soft <-> hard selection: train soft with annealed eps and decide hard at inference (MH34, MH41), or hard forward with a surrogate backward (MH46).
- Adapter: NMS / aggregation <-> end to end: allowed after training (MH12 uses NMS even in a learned system), or replaced by learned sibling rescoring (MH31, MH32, MH26 diagnosis).
- Adapter: masking <-> pruning (MH48).
- Conflict: Hungarian-based variants (MH23 - MH26, MH28, MH55) with the project rule; only the ideas transfer (K positives per target in training, one set at inference).
- Conflict: scene-level joint WTA (MH14) couples all agents; tips are independent.
- Conflict: soft or OT selection that blends positions (MH41, MH44); entmax with a fork unless the target is two-hot (MH35); normalised scores as validity (MH01).
- Conflict: RL / policy gradient (MH40, MH52) with "no argmax on the training path".
- Conflict: detaching across levels (MH53, MH14) removes the path from a downstream loss to the early children (cf. the E121 finding: planner loss through positions gives no gain).

### 3.6 Pairwise flags for the synthesis

| pair | flag | reason / evidence |
|---|---|---|
| nearest-child WTA x BCE score head | direct | MH01, MH02 (aMCL reuses the rMCL scoring), MH10 |
| WTA x annealing / eps floor / catch-up | direct | MH02, MH04, MH03 |
| WTA x on-lane validity | adapter | rMCL scores are cell mass; add a validity head (MH29) |
| WTA x point-to-polyline distance | adapter | step length unpinned (3.5) |
| WTA x two GT branches | adapter | independent nearest assignment can merge both branches into one child; ordered assignment [own inference] |
| anchors (MH10, MH12, MH16) x ray fan | adapter | bearings x ranges are the anchor set; top-b after suppression |
| dense pool + quality argmax (MH19 - MH21) x WTA | adapter | argmax per GT is WTA over the pool with a prediction-aware metric |
| learned duplicate removal (MH31, MH32) x beam | adapter | rescoring before the top-M cut; label = best candidate per GT branch |
| hard top-M x local score supervision | direct | all detectors |
| hard top-M x gold-in-beam | direct | MH42, MH49, MH50 |
| soft top-M x teacher forcing mix | direct | MH34 (rho = 0.8), MH41 (cross-entropy warm start) |
| soft / OT selection x node positions | conflict | blended positions leave the lane (MH41, MH44) |
| entmax x fork | conflict unless two-hot | MH35 |
| normalised scores x independent validity | conflict | different semantics (3.1) |
| Hungarian variants x project | conflict | project rule |
| NMS / aggregation x end to end | adapter | after training, or learned rescoring |
| free roll-out x multi-hypothesis | adapter | MH53 (one executed mode, detach) |
| free roll-out from step 1 x any | unknown | all precedents warm up with teacher forcing or ramp (MH41, MH49, MH51, MH17) |
| annealing x on-policy rollout | unknown | no paper |
| entmax / sparsemax x training-time beam | unknown | no paper (MH35 uses a beam only for decoding) |
| denoising (MH27) x hypothesis generation | adapter | trains refine / rescore with known targets, not the generation of new hypotheses |

### 3.7 Combinations the papers themselves demonstrate

WTA + BCE score head (MH01); annealed WTA + the same score head (MH02); eps-WTA and top-n-WTA combined with rMCL (MH01 experiments); EWTA + mixture fitting + NLL fine-tuning (MH05); learned anchors + hard WTA + likelihood aggregation + ensembling (MH11); static intention queries + dynamic queries + NMS + dense auxiliary loss (MH12); 64 learned queries + WTA + aggregation to 6 (MH13); dense candidates + heat-map CE + learned set predictor + offline optimiser (MH15); k-means anchors + truncated diffusion + per-anchor BCE on a BEV backbone (MH16); policy over lane-graph edges + latent z + WTA regression + K-means (MH17); quality-based one-to-one assignment + 3D max filter + auxiliary one-to-many (MH19); dual one-to-many / one-to-one heads with one metric (MH21); positive queries + auxiliary heads (MH25); DAC-DETR + Align-DETR loss 50.9 AP (MH26); soft top-k beam + teacher-forcing mix rho = 0.8 (MH34); entmax + exact beam decoding (MH35); gold alignments added to the beam normaliser (MH42); margin loss + beam reset + cross-entropy warm start + beam-size curriculum (MH49); pre-generated tree + conditional prediction + learned cost trained jointly (MH57); one executed WTA mode + detached closed-loop training (MH53). Not demonstrated anywhere found: WTA or anchors inside a beam over levels; entmax in a training-time beam; annealing with on-policy rollouts; any of these on geometric tree nodes.

### 3.8 Hyper-parameter values seen (starting points, not recommendations)

eps = 0.05 (MH04); T0 = 0.5, rho = 0.95 per epoch, 1000 epochs (MH02); beta = 1 for catch-up and scoring losses (MH01, MH03); k winners from M to 1 (MH05); 16 - 64 anchors (MH10), 64 intention points to 6 with a 2.5 m NMS (MH12), 64 modes to 6 with D = 2.3 / 2.9 (MH13), 8 learned modes (MH18), 20 / 18 anchors (MH16); group count K = 11 (MH23), repeats K = 6 and T = 1500 or 300 queries (MH24, MH28); alpha = 0.8 in POTO quality (MH19), alpha = 0.25, tau = 1.5 in Align-DETR (MH29); beam 3 with alpha annealed (MH41), beam 500 (MH42); SOFT eps 0.1 in beam search (MH34); entmax alpha 1.33 - 1.5 (MH35); soft resampling alpha 0.1 (MH44), OT eps 0.5 (MH44); Gumbel-softmax tau 1.0 -> 0.5 (MH46); load balance 0.01 (MH47); scheduled sampling per-token coin with a decaying probability (MH51); on-policy rollout step 2.0 s of a 6 s horizon (MH53).

## 4. Reuse versus re-implement

Environment: PyTorch 2.8, no mmcv / mmdet / mmdet3d / mmengine, one RTX 3060, cached 200 x 200 BEV pyramid, vertex + token + validity + edge output.

- Re-implement (effort S each, from the method sections): the WTA family with a BCE winner score (MH01 - MH05, about 30 lines); anchor / slot classification with nearest assignment (MH10 - MH12; the E94 fan is already such an anchor set); quality-argmax labels (MH19, MH20) and the IoU-aware target (MH29); a Relation-Network-style duplicate classifier with rank embedding (MH31, about 0.3 M parameters in the paper); sparsemax and 1.5-entmax (MH35, a sort-based routine; the entmax package is not on the environment list); Sinkhorn soft top-k (MH34); perturbed top-k (MH37); SIMPLE dynamic programme (MH38); soft resampling / stop-gradient reweighting (MH43, MH45); straight-through and Gumbel-softmax (MH46; PyTorch has the Gumbel-softmax function, from memory); beam-aware losses (MH49, MH50); scheduled-sampling coin (MH51); one-mode closed-loop rollout with detach (MH53); attention-mask pruning (MH48).
- Adapt code: nothing recommended. The DETR-family repositories (MH23 - MH29) build on frameworks absent here (mmdet / detectron2 / detrex style, from memory); DeFCN is Detectron2-based (MH19); MTR needs custom operators (from memory) and a token interface. The aMCL repository (MH02) is a readable PyTorch reference for the losses but the losses are a few lines.
- Not suitable as published: Hungarian variants (MH23 - MH26, MH28, MH55); RL and policy-gradient routes (MH40, MH52); anything needing perspective image features.
- Baselines worth keeping for comparisons: hand NMS top-b of the ray endpoints with a 2 - 3 m radius (MH12 style); dense pool + quality argmax (MH19 / MH20); plain WTA with a BCE winner score (MH01); rank-score MHT (MH56) as the non-learned counterpart.

## 5. Open questions and gaps

1. No paper trains a b-ary geometric tree with a beam through the selection. Closest: pre-generated tree with learned cost and rule pruning (MH57), Hungarian-matched tree tracker (MH55), sampled lane-graph traversals (MH17), mode slot held fixed along a rollout (MH18).
2. WTA with b >= 2 slots and mostly single-GT inputs: the remedies (MH02 - MH05) were demonstrated with at least 3 hypotheses on regression tasks; there are no numbers for the two-slot case and none for how fast the second slot specialises on rare forks.
3. Calibration of products of scores along a tree (what a beam ranks by) is unstudied; leaderboard Brier-style metrics exist only for flat trajectory sets.
4. Gradient to dropped nodes is demonstrated only on discrete or latent states (MH41, MH34, MH44); for geometric nodes the blending of positions is unaddressed.
5. Free roll-out with several hypotheses: only MH53 (receding-horizon replanning), no tree-wide result; all sequence precedents warm up with teacher forcing or ramp.
6. Hungarian-free assignment in crowded configurations: min-cost labelling fails on CrowdHuman (MH20), the remedy is bipartite matching; parallel lanes and junction mouths are the analogue and no matching-free remedy was found.
7. Entmax / sparsemax inside a training-time beam and with two-hot fork targets: not found.
8. Not read: the original MCL paper (algorithmic details only via sMCL, MH06), QCNet itself, Ding et al. 2022 (MH33), Kim et al. MHT papers (MH58), the details of DynamicViT (MH48), the OTA dynamic-k rule (MH22), the last parts of several long pages (marked in the cards).
9. No learned or differentiable multiple hypothesis tracker with a trainable hypothesis tree was found (MH58 search).
10. All numbers are transcribed from page summaries; where the summary corrected my question (for example Goyal et al. have no NMT experiment, Collobert et al. use WSJ, DAC-DETR gives no k) the card follows the page.

## 6. Search log

- Multiple Choice Learning Guzman-Rivera Batra Kohli NIPS 2012 -> listing + abstract; PDFs (Microsoft, NeurIPS) not decodable.
- arXiv abs 2311.01052 (rMCL), 2407.15580 (aMCL); full text ar5iv 2311.01052, arxiv.org/html 2407.15580; PDF fetch of 2311.01052 not decodable.
- sMCL arXiv 1606.07839 (abs + ar5iv); Confident MCL ar5iv 1706.03475; relaxed WTA ar5iv 1612.00197; EWTA ar5iv 1906.03631; DiverseNet ar5iv 2008.10634 (after a search for the title); Letzelter 2024 abs 2406.04706.
- Trajectory heads: arXiv HTML 2209.13508 (MTR), 1910.05449 (MultiPath), 2111.14973 (MultiPath++), 2207.05844 (Wayformer), 2306.10508 (QCNeXt report; QCNet CVF PDF 403), ar5iv 2106.08417 (Scene Transformer), ar5iv 2108.09640 (DenseTNT) after a search for TNT / DenseTNT, arXiv HTML 2411.15139 (DiffusionDrive), ar5iv 2106.15004 (PGP), arXiv HTML 2403.07232 (Villaflor).
- Search: "tree-structured policy planning learned behavior models" -> TPP 2301.11902, DTPP; arXiv HTML 2310.05885 (DTPP).
- Search + fetch for diversity: ar5iv 1907.04967 (DPP), ar5iv 1610.02424 (Diverse Beam Search), arXiv HTML 2411.19747 (multi-loss).
- DETR assignment: search "Group DETR ... 2207.13085"; ar5iv 2207.13085, 2207.13080 (H-DETR), 2211.12860 (Co-DETR), 2203.01305 (DN-DETR), 2304.07527 (Align-DETR); search "DAC-DETR ..." + text copy at object.cloud.sdsc.edu; arXiv HTML 2308.05736 (MapTRv2); abs 2212.06137 (DETA); search "O2SFormer ..." -> 2305.00675.
- Hungarian-free detectors: ar5iv 2012.03544 (DeFCN / POTO), ar5iv 2012.05780 (OneNet), arXiv HTML 2405.14458 (YOLOv10), ar5iv 2107.08430 (YOLOX / SimOTA).
- Duplicate removal: search "Ding Gall GCPR 2022 ..." (twice; no arXiv version); GCPR PDF not decodable; ar5iv 1711.11575 (Relation Networks), 1511.06437 (Tnet), 1705.02950 (GossipNet).
- Differentiable selection: ar5iv 1708.00111 (soft beam), 1902.06022 (differentiable beam decoder), 1903.06059 (Stochastic Beams), 2002.06043 (unordered-set estimator), 2002.06504 (SOFT top-k), 1905.11885 (OT sort), 2002.08676 (perturbed optimizers), 1901.10517 (relaxed subset sampling), 2210.01941 (SIMPLE), 1905.05702 (entmax), 1602.02068 (sparsemax).
- Particle filters and resampling: ar5iv 1805.11122 (Jonschkowski), 1805.08975 (Karkus), 2102.07850 (Corenflos), 2106.10314 (Scibior-Wood).
- Straight-through: ar5iv 1611.01144 (Gumbel-softmax), 1903.05662 (Yin), 2304.08612 (ReinMax).
- Routing and pruning: ar5iv 2101.03961 (Switch); ar5iv 2106.02034 and 2207.01580 (DynamicViT; conversion failed); search "DynamicViT ... Gumbel-softmax attention masking".
- Beam-aware training and exposure bias: ar5iv 1606.02960 (BSO), 1811.00898 (wrong paper, discarded), 1811.00512 (Negrinho), 1506.03099 (scheduled sampling), 1803.09693 (Polygon-RNN++); search "Negrinho Gormley Gordon ..."; search "multimodal trajectory prediction closed-loop ... exposure bias" -> arXiv HTML 2603.23393 (Yadav), abs 2603.24155 (withdrawn); aclanthology 2024.lrec-main.796 (El Khbir).
- Tracking and trees: search "learned multiple hypothesis tracking differentiable ..." (twice) and "Kim Li Rehg ... neural gating"; CVF PDF for Kim 2015 (403); search "tree-structured decoder hypothesis tree ... vessel centerline tracking" -> Selvan ar5iv 1806.08981, RefTr abs 2511.20823; search "Trexplorer ..." -> arXiv HTML 2507.10881; MICCAI PDF not decodable.
- Gap searches: "road graph centerline tracing multiple next-vertex hypotheses winner-takes-all ..." (extended); "differentiable beam search over geometric hypotheses ... lane graph or vessel tree tracing" (extended).
- Code checks: searches for rMCL and aMCL repositories; neurips.cc poster page 94844 (aMCL project link); GitHub README of Victorletzelter/annealed_mcl; a mistaken poster URL for rMCL returned another paper and was discarded.
