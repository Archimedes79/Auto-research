# Cluster VS - verification along a line, sampling patterns and feature access, refinement cascades

Literature research for the compatibility space, loop 16, date 2026-10-23. Card prefix VS. Judged against `PROJECT_CONTEXT.md` (cached 200 x 200 BEV pyramid, ego metres, vertex + token + validity + edges, matching-free losses, WSL env without mmcv / mmdet, RTX 3060).

## 1. Coverage statement
- Method: WebSearch and WebFetch only (no downloads, no git, no pip). About 100 tool calls. Pages were read through WebFetch, which returns an extraction answered by a small model against the page text, not the verbatim page; numbers below are as returned, and wherever the extraction or the paper flagged an inconsistency (text vs table, two versions) it is said on the card. (The fetch tool saved three PDF responses as binary files in the session's tool-results folder by itself; nothing else was written to disk apart from this file.)
- Bibliographic details (author lists, venues, years) are given as they appeared on the pages; where a page did not show them (ar5iv renderings usually do not) they are standard citations written from memory and are not marked card by card.
- Depth M (method section read): hazard / survival (Nnet-survival, Kvamme-Borgan, DRSA, PonderNet), ray rendering (NeRF, Mip-NeRF 360 distortion loss, UniPAD, RenderOcc, Vampire), line verification (L-CNN, HAWP, LETR, ELSD, DeepLSD, ScaleLSD), cascades and refinement (Cascade R-CNN, DINO, Deformable DETR, DAB-DETR, Sparse R-CNN, RAFT, Sparse4D v1 / v3, DAT), lane sampling (CLRNet, LaneATT, Polar R-CNN, BezierLaneNet, BezierFormer, CondLaneNet, UFLD, GANet, Anchor3DLane, LaneSegNet, TopoBDA about 90 % of the page, CurveFormer, LATR), polar (PolarMask, PolarFormer, PolarBEVDet), FPN level rule, Occ3D visibility masks.
- Depth M+C (code file also read as a web page): Deformable DETR (`ms_deform_attn.py`, `ms_deform_attn_func.py`, `main.py`), RAFT (`corr.py`, `raft.py`), CLRNet (`clr_head.py`), pycox (`loss.py`, with the hazard papers). Code pages for the DeepHit authors' repository (`class_DeepHit.py`) were read, but the DeepHit paper itself only at A.
- Depth A only: DeepHit paper (AAAI abstract; loss read in the authors' code), D-calibration (Haider et al., abstract), Khurana et al. differentiable raycasting (abstract and code README; the method section could not be opened: arXiv and ECVA PDFs came back as binary, ar5iv conversion failed; the raycasting equations were read in the successor paper Agro et al. 2023, depth M, and are attributed to that paper), the parametric-depth BEV paper (abstract: "visibility-aware evaluation metric", how visibility is obtained not stated), the polar adaptive BEV paper and the GaussianCaR outputs (search-result summaries only), ACT (only through PonderNet's comparison), TopoCurve (abstract on the poster page). PanopticBEV: occlusion-label section read (first 100k of 110k characters).
- Not found: any perception paper that uses a cumulative survival product along a ray with right censoring, nor one that selects endpoints from it; any lane / road method with ego-centred polar rays in BEV; any method that picks the pyramid level by distance from the ego for centre-lines; Hermite-spline attention (only Bezier); HAWPv2 / v3 paper pages (one wrong arXiv id, not retried); hazard-calibration studies outside medicine.
- Two arXiv ids tried first were wrong (Anchor3DLane, HAWPv3) and were corrected or dropped; this cost calls but does not affect the content.

## 2. Cards
Order: hazard / survival, ray rendering, line verification, cascade and refinement, lane-specific sampling, polar, auxiliary (level rule, visibility, lane-extent synthesis). "(from memory)" = not seen on a page in this session.

Card index (regime and differentiability use the schema vocabulary; fit = fit_to_us / effort):
| id | technique | stage | depth | regime | differentiable | fit |
|---|---|---|---|---|---|---|
| VS01 | logistic hazard, right censoring (Nnet-survival, Kvamme-Borgan) | S3 L | M+C | matching_free | yes | reimplement S |
| VS02 | DRSA recurrent hazard + status CE | S3 L | M | matching_free | yes | reuse_idea S |
| VS03 | DeepHit PMF + ranking | S3 L | A+C | matching_free | yes | reuse_idea S |
| VS04 | PonderNet halting law, expected loss, geometric KL (ACT) | S3 L | M / A | matching_free | yes in training | reuse_idea S |
| VS05 | D-calibration | E | A | none | n/a | reuse_idea S |
| VS06 | NeRF volume rendering, transmittance | S3 X | M | self_supervised | yes | reuse_idea S |
| VS07 | Mip-NeRF 360 distortion loss | L S3 | M | self_supervised | yes | reuse_idea S |
| VS08 | UniPAD rendering pre-training | S3 B T | M | self_supervised | yes | not_suitable (idea S) |
| VS09 | RenderOcc rays, auxiliary rays, weighted sampling | S3 L | M | matching_free | yes | reuse_idea S |
| VS10 | Vampire BEV rendering along height | S3 B L | M | matching_free | yes | not_suitable (idea S) |
| VS11 | Khurana differentiable raycasting (equations via Agro et al.) | S3 S10 L | A+C (M via successor) | self_supervised | yes | not_suitable (idea S) |
| VS12 | L-CNN LoI verification | S3 S4 L | M | matching_free | partial (nms, sampler) | baseline S |
| VS13 | HAWP | S3 S4 | M | matching_free | partial | reuse_idea S |
| VS14 | DeepLSD inlier-fraction verification | S3 S5 | M | self_supervised | partial | reuse_idea S |
| VS15 | ScaleLSD support degree | S3 | M | self_supervised | partial | reuse_idea S |
| VS16 | LETR | S4 S5 | M | hungarian_set | partial | not_suitable |
| VS17 | ELSD centerness scoring | S3 S2 | M | matching_free | yes | reuse_idea S |
| VS18 | Cascade R-CNN | S5 L | M | matching_free | partial | reuse_idea S |
| VS19 | DINO look-forward-twice, mixed query selection, CDN | S5 S4 T | M | hungarian_set + denoising | partial | reuse_idea S |
| VS20 | Deformable DETR (+ code) | S2 S5 | M+C | hungarian_set | partial | reuse_idea / reimplement S |
| VS21 | RAFT lookup + update (+ code) | S2 S5 | M+C | matching_free | yes (stop-gradient) | reuse_idea S |
| VS22 | Sparse R-CNN | S5 S2 | M | hungarian_set | partial | reuse_idea S |
| VS23 | DAB-DETR | S2 S5 | M | hungarian_set | partial | reuse_idea S |
| VS24 | Sparse4D v1 / v3 keypoints, quality head | S2 S3 S8 | M | hungarian_set + denoising | partial | reuse_idea S |
| VS25 | DAT bounded offsets | S2 | M | none | yes | reuse_idea S |
| VS26 | CLRNet (+ code) | S2 S5 L | M+C | matching_free (dynamic top-k) | partial | reuse_idea / reimplement S-M |
| VS27 | LaneATT | S2 S4 | M | matching_free | yes except nms | reuse_idea S |
| VS28 | Polar R-CNN | S1 S2 S4 | M | matching_free (SimOTA) + hungarian_set (O2O) | partial | reuse_idea S-M |
| VS29 | BezierFormer | S2 S5 L | M | matching_free (SimOTA top-1) | partial | reuse_idea / reimplement M |
| VS30 | BezierLaneNet | S2 S4 L | M | hungarian_set | partial | reuse_idea S |
| VS31 | LaneSegNet lane attention | S2 S5 S6 | M | hungarian_set | partial | reuse_idea / reimplement M |
| VS32 | TopoBDA (camera + radar) | S2 S5 B | M | hungarian_set | partial | reuse_idea S |
| VS33 | CondLaneNet | S1 S2 S3 | M | matching_free (dense) | yes except peak picking | reuse_idea S |
| VS34 | UFLD | S2 S3 | M | matching_free | yes | reuse_idea S |
| VS35 | GANet | S2 S6 | M | matching_free + auxiliary hungarian | yes except grouping | reuse_idea S |
| VS36 | Anchor3DLane | S1 S2 S5 | M | matching_free | yes except nms | reuse_idea S |
| VS37 | CurveFormer | S2 S5 | M | hungarian_set | partial | reuse_idea S |
| VS38 | LATR | S2 S5 | M | hungarian_set | partial | reuse_idea S |
| VS39 | PolarMask | S2 L | M | matching_free | yes | reuse_idea S |
| VS40 | PolarFormer | B S2 | M | hungarian_set | partial | not_suitable (idea S) |
| VS41 | PolarBEVDet | B S8 S2 | M | matching_free | yes | reuse_idea S |
| VS42 | FPN level rule | S2 | M | none | yes | reuse_idea S |
| VS43 | visibility / observability masks | S3 D E | M / A | labels | n/a | reuse_idea S |
| VS44 | lane extent heads (synthesis) | S3 S7 | M | - | - | reuse_idea S |

### CARD VS01 - Discrete-time hazard network with right censoring (logistic hazard = Nnet-survival)
- papers: A scalable discrete-time survival model for neural networks (Gensheimer, Narasimhan, PeerJ 2019, arXiv 2018) https://arxiv.org/abs/1805.00917 ; Continuous and discrete-time survival prediction with neural networks (Kvamme, Borgan, arXiv 2019; journal version from memory) https://arxiv.org/abs/1910.06724 ; pycox implementation https://github.com/havakv/pycox
- depth: M+C (pycox `loss.py` read)
- stage: S3, L
- idea: Cut the "time" axis (for us: distance along a ray) into intervals. One sigmoid per interval gives the conditional hazard h_j = P(fail in interval j | alive at its start); survival through j is S_j = prod_{i<=j}(1 - h_i). Training is binary cross-entropy over the intervals up to the observed one; a censored case just stops contributing after its last observed interval. This is exactly the algebra of our survival product (nu_k = 1 - h_k).
- formulation: event in interval k: log L = log h_k + sum_{j<k} log(1 - h_j); censored: sum of log(1 - h_j) over the survived intervals. The censoring-interval CONVENTION differs between sources: Nnet-survival credits a censored case with the interval of censoring only if the censoring time lies in its second half (labels surv_s / surv_f; loss = sum_i [ln(1 + surv_s(i)(p(i) - 1)) + ln(1 - surv_f(i) p(i))], p = predicted conditional survival); Kvamme-Borgan and pycox `nll_logistic_hazard` count the censoring interval as survived (BCE over j = 0..idx inclusive, target 1 only at the event interval); pycox `nll_pmf` (PMF form) excludes it (mass strictly after); pycox `nll_pc_hazard` (hazard = softplus(phi), piecewise constant) weights the last interval by the observed fraction `interval_frac`. Intervals: 15 - 40 in most Nnet-survival experiments (19 for SUPPORT), spaced wider late so every interval holds a similar number of events; Kvamme-Borgan simulated grids of 5 / 25 / 100 / 250 points, the number of intervals must be tuned (fine grid: less bias, more variance), Kaplan-Meier-quantile grid slightly better than equidistant, constant-density or constant-hazard interpolation inside an interval reduces sensitivity to the grid. Nnet-survival also offers a proportional-hazards variant (one output, 1 - h_i = (1 - h_base)^exp(X beta)).
- input: point_set of per-interval features (our use: ring samples from bev_raster + context). output: scores (1 - h_j per interval).
- training: likelihood = masked BCE; labels (interval of event or censoring, event flag); matching_free.
- differentiable: yes (sigmoid, sums, products; no argmax).
- sensors_data: tabular medical data (SUPPORT, METABRIC, FLCHAIN, NWTCO, GBSG), MNIST survival; nothing spatial.
- results: Nnet-survival SUPPORT test (n = 2,732): C-index 0.732, Brier at 6 mo / 1 y / 3 y 0.181 / 0.184 / 0.177 (Cox PH 0.734; 0.183 / 0.186 / 0.178); calibration shown only qualitatively (predicted and Kaplan-Meier curves "superimposed" in the simulation). Kvamme-Borgan 5-fold concordance, Logistic-Hazard: FLCHAIN 0.792, METABRIC 0.658, SUPPORT 0.625 (PMF 0.786 / 0.632 / 0.627); integrated Brier Logistic-Hazard 0.0918 / 0.172 / 0.213; PMF slightly worse and less stable across grids (paper text).
- compute: negligible (n sigmoids).
- code: checked https://github.com/havakv/pycox (loss.py); Nnet-survival repo https://github.com/MGensheimer/nnet-survival (URL from memory, not opened)
- fit_to_us: reimplement - S - ring gaps are unequal (4 / 4 / 4 / 6 / 6 m) and the convention at the BEV edge or in occlusion must be fixed (notes 3b).
- compat_needs: ordered samples per ray; labels for sample k only while all earlier samples were alive (BCE masked after the first failure); a censoring flag per ray (range or occlusion).
- compat_conflicts: one failure per ray - a ray that leaves the lane and re-enters (occluded stretch, dashed or crossing lanes) violates the single-event model; a marginal "near lane" label at every sample is NOT the conditional hazard.
- tried_before: none (E94 predicts events continue / end / fork / merge and a step class, not a survival product).
- verified: checked

### CARD VS02 - DRSA: deep recurrent survival analysis (hazard per slot + survival-status cross-entropy)
- papers: Deep Recurrent Survival Analysis (Ren et al., AAAI 2019) https://arxiv.org/abs/1809.02403
- depth: M
- stage: S3, L
- idea: A recurrent cell emits a conditional hazard per time slot, so the hazard at slot l depends on earlier slots. Besides the event-time likelihood, a cross-entropy on the survival STATUS at the observed time supervises the whole range before the event; the authors say this gives denser gradients than a pure event-time likelihood (DeepHit touches mainly the true event time).
- formulation: h_l = Pr(z in V_l | z > t_{l-1}, x) from an LSTM cell fed (x, t_l, previous hidden state); S(t) = prod_{l<=l^i}(1 - h_l); event probability p_l = h_l prod_{j<l}(1 - h_j). L_z = -log p at the observed slot (uncensored samples only). L_c = survival-status cross-entropy: uncensored -log(1 - S(t^i)), censored -log S(t^i). Objective alpha L_z + (1 - alpha) L_c (Eq. 17); alpha value not stated; whether the hazard output is a sigmoid is not stated on the page seen.
- input: point_set / features per slot. output: scores (hazard per slot).
- training: likelihood + status CE; matching_free.
- differentiable: yes.
- sensors_data: CLINIC (6,036 samples), MUSIC (3,296,328, last.fm return times), BIDDING (19,495,974, auction win times); nothing spatial.
- results: C-index CLINIC / MUSIC / BIDDING: DRSA 0.774 / 0.892 / 0.911, DeepHit 0.733 / 0.878 / 0.858, DeepSurv 0.753 / 0.862 / 0.840; ANLP 3.337 / 5.132 / 4.774 (DRSA) vs 5.027 / 5.523 / 5.544 (DeepHit) (paper Table 2; Lasso-Cox, not plain Cox, is the Cox baseline).
- compute: not stated.
- code: URL from memory, not opened: https://github.com/rk2900/DRSA
- fit_to_us: reuse_idea - S - our per-ring on-lane labels already act as the status loss; an optional recurrence along the ring index (GRU over 4 / 8 / 12 / 18 / 24 m) makes the hazard depend on earlier samples.
- compat_needs: ordered samples; a recurrent or attention link over the ring index if sequential dependence is wanted.
- compat_conflicts: none for K = 5 rings (sequential compute is tiny).
- tried_before: none.
- verified: checked

### CARD VS03 - DeepHit (single risk): PMF over (time) cells + ranking loss
- papers: DeepHit (Lee, Zame, Yoon, van der Schaar, AAAI 2018) https://ojs.aaai.org/index.php/AAAI/article/view/11842 ; authors' code https://github.com/chl8856/DeepHit ; re-implementation https://github.com/havakv/pycox
- depth: A (paper abstract) + C (authors' `class_DeepHit.py` and pycox `loss.py` read)
- stage: S3, L
- idea: Instead of a hazard per interval the network outputs ONE softmax over all (event, time) cells = the probability mass of the event time; a pairwise ranking loss orders individuals by risk.
- formulation: (authors' code) shared sub-network with residual input concatenation, cause-specific sub-networks, dropout, linear layer to num_Event x num_Category logits, one softmax over all cells. LOSS_1 log-likelihood: uncensored = log probability of the observed cell; censored = log of the summed mass after the censoring time (= survival past it). LOSS_2 ranking: for subject pairs with t_i < t_j and event for i, penalty exp(-(F_i(t_i) - F_j(t_i)) / sigma1), sigma1 = 0.1 fixed in the file. LOSS_3 calibration (squared error of batch-summed predicted probability against the event indicator) exists in the code only; a comment says it is not in the paper version. Total = alpha LOSS_1 + beta LOSS_2 + gamma LOSS_3 + L2 (1e-4, hidden) + L1 (1e-4, output); alpha, beta, gamma are passed at training time, no defaults in the file. pycox single-risk version: nll_pmf + rank loss, alpha-weighted.
- input: features. output: scores (PMF over cells + tail).
- training: likelihood + ranking; matching_free.
- differentiable: yes.
- sensors_data: medical tabular (paper not read).
- results: none seen for the original paper; Kvamme-Borgan (VS01) report PMF slightly worse than logistic hazard (concordance and IBS above).
- compute: not stated.
- code: checked (two code pages above).
- fit_to_us: reuse_idea - S - a direct softmax over {endpoint at sample 0..K} is the "PMF" alternative to the hazard parametrisation: P(k) without a product, but no per-sample interpretation and, per Kvamme-Borgan, less stable across grids.
- compat_needs: none beyond VS01.
- compat_conflicts: the ranking loss needs comparable pairs (pairs of rays); not needed here.
- tried_before: none.
- verified: partly (paper not opened)

### CARD VS04 - PonderNet (halting hazard, expected loss under the halting law) and ACT
- papers: PonderNet: Learning to Ponder (Banino, Balaguer, Blundell, arXiv 2021) https://arxiv.org/abs/2107.05407 (HTML v2 read) ; Adaptive Computation Time (Graves, 2016, from memory) https://arxiv.org/abs/1603.08983 (not opened)
- depth: M (PonderNet), A (ACT)
- stage: S3, L
- idea: A network run for a variable number of steps predicts at each step a halting probability lambda_n. The probability to stop at step n is p_n = lambda_n prod_{j<n}(1 - lambda_j) = the same hazard / survival algebra as our ray. The loss is the EXPECTED task loss under p_n plus a KL term to a geometric prior, so every candidate stop step receives a gradient weighted by its probability and no argmax is needed.
- formulation: L = sum_n p_n * Loss(y, y_n) + beta * KL(p || geometric(lambda_p)); beta = 0.01 for parity, bAbI and PAI; lambda_p searched in (0,1] (parity, bAbI) and (0, 0.5] (PAI); lambda_p = 0.9 was the only parity setting where the method failed; N = 20 (parity), 10 (bAbI), search 7 - 10 (PAI), or N = smallest n with sum_{j<=n} p_j > 1 - eps, eps = 0.05. Mass left after N: renormalise p_n or give the remainder to the last step (p_N = 1 - sum_{n<N} p_n) = right censoring at N. Inference: sample Bernoulli(lambda_n) per step (or use the expectation); expected steps = sum n p_n. Versus ACT (authors' statement): ACT stops when the cumulative halting sum reaches 1 - eps, outputs a weighted average, has a biased cost gradient and a hard-to-set tau; PonderNet has unbiased low-variance gradients. How lambda_n is produced (sigmoid) and calibration of halting: not stated on the page.
- input: state per step. output: scores (halting probability per step) + prediction per step.
- training: expected loss + KL; matching_free (supervised).
- differentiable: yes in training (sum over all steps); inference by sampling is not.
- sensors_data: parity, bAbI, Paired Associative Inference; nothing spatial.
- results: task-specific, not copied.
- compute: all N steps are evaluated in training.
- code: none found officially (not checked).
- fit_to_us: reuse_idea - S - (a) endpoint loss as an expectation over P(k) of a per-endpoint regression loss; (b) a geometric-prior KL on P(k) as length prior against early stopping; (c) the leftover mass at the last ring = censoring at the range limit.
- compat_needs: a per-candidate loss that can be evaluated for every k (we can: distance of sample k to the GT lane).
- compat_conflicts: expected-loss training spreads mass over candidates; a sharp argmax endpoint at inference then needs calibration.
- tried_before: none.
- verified: checked

### CARD VS05 - D-calibration: evaluating individual survival distributions
- papers: Effective ways to build and evaluate individual survival distributions (Haider, Hoehn, Davis, Greiner, JMLR 2020) https://arxiv.org/abs/1811.11347
- depth: A
- stage: E
- idea: Argues for models that output a whole survival distribution per individual and for "D-calibration" to test whether those probabilities are meaningful; compares concordance, 1-calibration, Brier score and L1-loss variants and says they measure different things.
- formulation: not read (from memory, unverified: predicted survival probability at each individual's event time should be uniform on (0,1), with a rule for censored cases).
- input / output / training: not read. differentiable: not applicable.
- sensors_data: medical.
- results: none seen.
- compute: not stated. code: not checked.
- fit_to_us: reuse_idea - S - a probability-integral-transform style check of the endpoint law P(k) on the labelled rays (own suggestion; the discrete ring grid needs randomised ties).
- compat_needs: censoring handling in the evaluation. compat_conflicts: none.
- tried_before: none.
- verified: partly

### CARD VS06 - NeRF volume rendering: alpha compositing with transmittance
- papers: NeRF (Mildenhall et al., ECCV 2020) https://arxiv.org/abs/2003.08934
- depth: M
- stage: S3, X
- idea: A ray accumulates a per-sample quantity with weights w_i = T_i * alpha_i, where alpha_i is the chance that the ray is stopped inside interval i and T_i the chance that it was not stopped earlier. Hierarchical sampling re-draws samples where w is large.
- formulation: C(r) = integral T(t) sigma(r(t)) c dt with T(t) = exp(-integral_{tn}^{t} sigma); discrete: C^ = sum_i T_i (1 - exp(-sigma_i delta_i)) c_i, T_i = exp(-sum_{j<i} sigma_j delta_j), delta_i = t_{i+1} - t_i (= prod_{j<i}(1 - alpha_j) with alpha_i = 1 - exp(-sigma_i delta_i); the paper says it "reduces to alpha compositing" but does not write the product). T is interpreted as the probability that the ray travels without hitting anything. Stratified sampling: N bins, one uniform draw per bin. Hierarchical: N_c = 64 coarse, N_f = 128 fine by inverse-transform sampling from the normalised w_i of the coarse pass; the fine pass uses all 192; loss = squared colour error of coarse + fine. The appendix quotes 64 + (64 + 128) at test time (differs from the training count; not reconciled). Near / far bounds are symbols only (synthetic scenes inside a cube of side 2; forward-facing scenes in NDC with far = infinity). The last delta and rays that leave the volume are not discussed.
- input: density and attribute at sample points along a ray. output: rendered attribute, weights w_i.
- training: self_supervised (photometric); no matching.
- differentiable: yes (the stratified draws are re-sampled each step).
- sensors_data: multi-view RGB images.
- results: none copied (rendering paper).
- compute: 4,096 rays per batch, 192 network calls per ray in the fine pass (training).
- code: URL from memory, not opened: https://github.com/bmild/nerf
- fit_to_us: reuse_idea - S - map: alpha_i <-> hazard (1 - nu_i), T_i <-> survival, w_i <-> probability that the last verified sample is the one before i. Writing the per-sample probability as exp(-sigma_i delta_i) (hazard per metre times gap) keeps the model consistent when ring gaps differ (4 / 4 / 4 / 6 / 6 m) or are changed.
- compat_needs: a density along the ray; for us density = "lane absent".
- compat_conflicts: the remainder 1 - sum w_i (the ray never stopped) is ignored by the paper; for a censored ray it must be kept explicitly.
- tried_before: none.
- verified: checked

### CARD VS07 - Mip-NeRF 360 distortion regulariser (compact ray weights)
- papers: Mip-NeRF 360 (Barron et al., CVPR 2022) https://arxiv.org/abs/2111.12077
- depth: M
- stage: L, S3
- idea: A penalty on the ray weights w that pulls their mass into one short interval, so the endpoint distribution along a ray stays compact; plus a proposal-network loss that upper-bounds the fine weights with the coarse ones under stop-gradient.
- formulation: L_dist(s, w) = sum_{i,j} w_i w_j |m_i - m_j| + (1/3) sum_i w_i^2 (s_{i+1} - s_i), m_i = interval midpoint in NORMALISED ray distance s (metric t "significantly upweights distant intervals"); weight lambda = 0.01. Effects named by the authors: shrink each interval, pull distant intervals together, consolidate weight into one or a few neighbouring intervals, drive weights to zero where possible. L_prop = sum_i (1 / w_i) max(0, w_i - bound(t^, w^, T_i))^2, bound = sum of proposal weights overlapping T_i, asymmetric, stop-gradient on the NeRF outputs. The paper's w_i equals (1 - exp(-tau_i (t_{i+1} - t_i))) * exp(-sum_{i'<i} tau_{i'} (t_{i'+1} - t_{i'})) (Eq. 4).
- input: weights and interval borders along a ray. output: scalar loss.
- training: regulariser (self_supervised context).
- differentiable: yes.
- sensors_data: images.
- results: without L_dist "floaters" remain (qualitative figure); metrics change little.
- compute: O(K^2) per ray, trivial for K = 5.
- code: URL from memory, not opened: https://github.com/google-research/multinerf
- fit_to_us: reuse_idea - S - regulariser on P(k) against bimodal endpoint laws (the "hop to the neighbouring lane" failure); needs the metric distance normalised to 24 m on our non-uniform rings.
- compat_needs: P(k) as weights; normalised distances.
- compat_conflicts: penalises genuine bimodality (a fork inside one ray); only sensible if forks are covered by separate rays.
- tried_before: none.
- verified: checked

### CARD VS08 - UniPAD: differentiable rendering as 3D pre-training (SDF-based opacity along rays)
- papers: UniPAD (Yang et al., CVPR 2024) https://arxiv.org/abs/2310.08370
- depth: M
- stage: S3, B, T
- idea: Voxel features (lidar, camera or both) are rendered to RGB and depth along camera rays by a NeuS-style SDF volume renderer; the rendering loss pre-trains the backbone; block masking of the input is added.
- formulation: sample points p_j = o + t_j d, features by trilinear interpolation of a 180 x 180 x 5 voxel grid (32 channels); s_j = MLP_SDF(p_j, f_j); opacity alpha_j = max((sigma_s(s_j) - sigma_s(s_{j+1})) / sigma_s(s_j), 0), sigma_s(x) = 1 / (1 + exp(-s x)) with learnable sharpness s; T_j = prod_{k<j}(1 - alpha_k); w_j = T_j alpha_j; RGB = sum w_j c_j; depth = sum w_j t_j; 96 samples per ray; L1 colour and L1 depth (depth only on rays with lidar), both weight 10. Ray sampling: random (512 rays per view in the implementation details), depth-aware (pixels with lidar depth) or dilation (every I-th pixel): NDS 32.5 / 32.9 / 31.9, memory 1x / 1x / 1.4x. Masking: images block 32, ratio 0.3 (NDS peaks at 0.3: 32.9), points block 8, ratio 0.8. AdamW, 12 epochs.
- input: bev_raster-like voxel volume (lidar sparse conv or camera lift). output: pre-trained encoder.
- training: self_supervised (render RGB + depth), then fine-tune.
- differentiable: yes.
- sensors_data: nuScenes lidar + 6 cameras.
- results: nuScenes val detection NDS / mAP: UVTR-L 67.7 / 60.9 -> 70.6 / 65.0; UVTR-C 45.0 / 37.2 -> 47.4 / 41.5; UVTR-M 70.2 / 65.4 -> 73.2 / 69.9; the ablation Table 6 gains +9.1 NDS (lidar), +7.7 (camera), +6.9 (fusion) (the page notes headline and Table 1 baselines differ); segmentation SpUNet 73.3 -> 79.4 mIoU.
- compute: not stated beyond the memory ratios.
- code: URL from memory, not opened: https://github.com/Nightmare-n/UniPAD
- fit_to_us: not_suitable as published (3D voxels, camera rays, pre-training of a trainable encoder; our foundation is frozen); reuse_idea - S - sample rays only where a label exists (depth-aware) = verify only rays with an available lane label.
- compat_needs: trainable BEV / voxel encoder; camera ray geometry.
- compat_conflicts: frozen foundation; no pre-training stage.
- tried_before: none.
- verified: checked

### CARD VS09 - RenderOcc: 2D-supervised occupancy with volume rendering, auxiliary rays, weighted ray sampling
- papers: RenderOcc (Pan et al., ICRA 2024) https://arxiv.org/abs/2309.09502
- depth: M
- stage: S3, L
- idea: Semantics and depth are rendered from a 3D volume along camera rays and compared with 2D labels, so no 3D occupancy label is needed; rays from adjacent frames add views; a weighting scheme chooses which rays enter the loss.
- formulation: sigma = softplus(MLP_d(V)), S = MLP_s(V); alpha(z_k) = 1 - exp(-sigma(z_k) beta_k); T(z_k) = exp(-sum_{t<k} sigma(z_t) beta_t); semantics = sum_k T alpha S; depth = sum_k T alpha z_k. Loss = CE (semantics) + SILog (depth) + L_reg (distortion loss and total-variation loss on the density; weights not stated). One ray per pixel, 38,400 rays per batch on nuScenes (25,600 on SemanticKITTI); points per ray K not stated for the main model (ablation: unified, hierarchical 64 -> 64 or 64 -> 128, mip360-style step 0.5 voxel). Auxiliary rays: 6 adjacent frames on nuScenes (4 on SemanticKITTI) moved by ego pose. Weighted ray sampling: class-balance W_b = exp(lambda_s (max(M) / N(C(r)) - 1)) and a temporal weight (dynamic-object rays get a coefficient < 1), W = W_b W_t as sampling probability, fixed ray count per batch. Rays leaving the volume: not stated.
- input: volume feature V. output: occupancy / semantic volume.
- training: supervised from 2D labels (projected lidar with segmentation); matching_free.
- differentiable: yes (ray selection only chooses loss terms).
- sensors_data: nuScenes (Occ3D labels for evaluation), SemanticKITTI; cameras.
- results: Occ3D-nuScenes mIoU: RenderOcc 2D-only 23.93, with 3D labels too 26.11; BEVStereo 24.51, BEVFormer 23.67 (3D-supervised); depth from structure-from-motion instead of lidar 21.11 vs 23.93.
- compute: not stated.
- code: URL from memory, not opened: https://github.com/pmj110119/RenderOcc
- fit_to_us: reuse_idea - S - (a) supervise a per-sample ray accumulator with 1-D labels; (b) importance weights on which rays enter the loss (rare / hard rays); (c) the distortion regulariser (VS07).
- compat_needs: camera rays and a 3D density field.
- compat_conflicts: the cached BEV rasters carry no depth axis.
- tried_before: none.
- verified: checked

### CARD VS10 - Vampire: volume rendering in BEV along the height axis
- papers: Regulating intermediate 3D features for vision-centric autonomous driving ("Vampire", arXiv 2312.11837; authors not captured) https://arxiv.org/abs/2312.11837
- depth: M
- stage: S3, B, L
- idea: Besides rendering depth / semantics in camera view, a BEV feature volume is rendered "directly from the top-down height axis" into BEV height and BEV semantic maps; these rendering losses regularise intermediate features during joint training.
- formulation: density V_density = alpha * Psi_beta(V_sdf), Psi_beta = Laplace CDF with learnable alpha, beta > 0; T(t) = exp(-integral sigma) and w(t) = integral T sigma as printed; height D(r) = integral w r, semantics S(r) = integral w s; BEV volume 20 x 256 x 256 over height [-3, 5] m; maps rendered at 1/4 resolution; losses smooth-L1 (height / depth), CE + Lovasz (semantics), both weights 1; number of samples along the vertical axis not stated.
- input: bev_raster volume with a height axis. output: BEV height and semantic maps.
- training: auxiliary rendering losses in joint training (the paper does not call it pre-training); matching_free.
- differentiable: yes.
- sensors_data: nuScenes cameras (lidar labels for depth / height).
- results: nuScenes val ResNet-50, no temporal fusion: detection mAP 0.301, NDS 0.354 (BEVFormer 0.257 / 0.359 and BEVDet 0.286 / 0.372 as cited from Table 3); Occ3D-nuScenes mIoU 28.3 (appendix 28.33; the +1.4 vs BEVFormer in the text is 1.45 in the appendix); lidar segmentation mIoU 66.4 vs 59.3 (TPVFormer).
- compute: not stated.
- code: github.com/cskkxjk/Vampire (named in the abstract; release "upon publication" per appendix; not opened).
- fit_to_us: not_suitable (needs a trainable 3D / BEV volume); reuse_idea - S - the only BEV-plane example found: rays can be cast through a BEV feature volume, but along height, not along a lane.
- compat_needs: height-resolved volume. compat_conflicts: our cached features have no height axis.
- tried_before: none.
- verified: checked

### CARD VS11 - Differentiable raycasting for occupancy forecasting (Khurana et al.; formula read in the successor Agro et al.)
- papers: Differentiable raycasting for self-supervised occupancy forecasting (Khurana, Hu, Dave, Ziglar, Held, Ramanan, ECCV 2022) https://arxiv.org/abs/2210.01917 ; Point cloud forecasting as a proxy for 4D occupancy forecasting (Agro et al., CVPR 2023, venue from memory) https://arxiv.org/abs/2302.13130
- depth: A + C for Khurana et al. (abstract and code README; their method section NOT read); M for Agro et al. (ar5iv page), which extends the same raycaster to 3-D and states the equations itself
- stage: S3, S10, L
- idea: Predicted future occupancy is "rendered" into future lidar sweeps by differentiable raycasting; comparing with real sweeps trains the forecaster without occupancy labels; the emergent occupancy then drives a planner. In the successor the renderer is a hit-probability chain along each ray.
- formulation: (Agro et al., Eqs. 7 - 10; not attributed to Khurana by the page) probability that the ray stops at voxel i: p_i = prod_{j<i}(1 - z_j) * z_i with z the predicted occupancy in [0,1]; expected depth = sum_i p_i * lambda_i, lambda_i = distance at the voxel boundary where a ray stops; rays are traversed voxel by voxel (Amanatides-Woo), stops only at voxel boundaries or infinity; RAYS LEAVING THE GRID: in training a virtual stop at the observed depth beyond the grid is added (Eq. 9), whereas the appendix says the mass that does not terminate is placed at the grid boundary (the two descriptions differ), at evaluation rays are clamped to the volume; loss = L1 between rendered and measured depth summed over rays, gradients reach every voxel the ray crosses; grid = 4-D occupancy (X x Y x Z x T, Z and T folded into channels for 2-D convolutions), nuScenes +-70 m in x and y, +-4.5 m in z. Khurana et al.: README only - `rasterize.py` builds BEV object occupancy and "shadow" maps, `precast.py` future visible free-space maps and contour points; training arguments include sampled trajectories (clothoid or data-driven) and a weight on a free-space binary cross-entropy; test scripts compute ray-distance error and planning losses.
- input: past lidar sweeps. output: future occupancy / free space; rendered depth; planner cost (Khurana et al.).
- training: self_supervised from lidar; planner cost on occupancy.
- differentiable: yes (the point of the papers).
- sensors_data: lidar; nuScenes (with CAN bus) and ONCE named in the Khurana README; nuScenes, KITTI-Odometry, Argoverse 2.0 in Agro et al.
- results: Khurana abstract: raycasted-lidar forecasting up to 15 F1 points better; planner collisions up to 17 % fewer relative to freespace-centric planners. Agro et al. (their own numbers): nuScenes 1 s L1 1.40 m, AbsRel 10.37 %, 3 s L1 1.71 m, against a ray-tracing baseline 1.50 m / 2.44 m.
- compute: not stated.
- code: checked https://github.com/tarashakhurana/emergent-occ-forecasting (README: PyTorch, MIT licence stated on the page, C++ ground segmentation built with CMake, based on an earlier "ff" planner repo).
- fit_to_us: not_suitable (lidar); reuse_idea - S - the hit-probability chain p_i = prod(1 - z_j) z_i is again the survival algebra, and "virtual stop at the observed depth outside the grid" is a published way to give a ray that leaves the volume a defined endpoint (a censoring-like convention).
- compat_needs: lidar sweeps, voxel traversal. compat_conflicts: no lidar in our pipeline.
- tried_before: none.
- verified: partly (Khurana et al. method section not opened; the formulas come from the successor paper)

### CARD VS12 - L-CNN: junction proposals + line sampler + LoI verification
- papers: End-to-end wireframe parsing (Zhou, Qi, Ma, ICCV 2019) https://arxiv.org/abs/1905.03246
- depth: M
- stage: S3, S4, L
- idea: Junctions are detected on a 128 x 128 bin grid; every pair of the top-300 junctions is a line hypothesis; each hypothesis gets ONE score from a verification network that sees 32 bilinearly sampled feature vectors along the segment (line-of-interest pooling).
- formulation: LoI: N_p = 32 points q_k between the endpoints (linear interpolation), bilinear read of the backbone feature (256 ch reduced by a 1x1 conv to 128), 1-D max-pool stride 4 (32 -> 8 points), flatten 128 x 8, FC(1024) + ReLU, FC -> logit; the feature-map size used for pooling is not stated (junction grid 128 x 128, input 512 x 512). Loss: sigmoid BCE, mean over positives plus mean over negatives per image (balance); multi-task weights junction 8, offset 0.25, verification 1. Junction head: bin likelihood BCE + offset L2 in [-1/2, 1/2); 3x3 max-pool NMS; top K = 300. Sampler: static (300 positives from GT lines, 40 hard negatives from 2,000 pairs ranked by density of the rasterised GT) and dynamic from predicted junctions (match threshold eta not stated): D+ 300, D- 80, D* 600 per image. Inference: all pairs of 300 junctions. Adam 4e-4, wd 1e-4, batch 6, 16 epochs (about 8 h on one GTX 1080 Ti), stacked hourglass.
- input: bev_raster-like feature map + point_set of junction pairs. output: segments with scores.
- training: BCE on thresholded junction matches; matching_free.
- differentiable: partial (junction NMS / top-K and the sampler are discrete; the verifier is differentiable in the features).
- sensors_data: images (ShanghaiTech wireframe, York Urban).
- results: ShanghaiTech sAP10 62.9, APH 82.8, FH 81.2; York Urban sAP10 26.4, APH 59.8 (Table 2); post-processing lifts FH 76.9 -> 81.2. Ablation facts seen: N_p was NOT ablated; a 1-D conv instead of the FC head lowers sAP5 57.4 -> 55.7; hand-made coordinate / slope features do not help; hard-negative samplers help. FPS not stated (HAWP reports 15.6 FPS for a re-trained L-CNN).
- compute: not stated.
- code: URL from memory, not opened: https://github.com/zhou13/lcnn
- fit_to_us: baseline - S - single-score verifier on the same teardrop ray (the earlier review demanded this comparison); pair-then-verify is also the published hypothesis-selection precedent.
- compat_needs: both endpoints known before verification (junctions); fixed 32 samples whatever the length.
- compat_conflicts: one score per line cannot say WHERE a long line stops being a lane; our ray's endpoint is unknown.
- tried_before: E116 junction / piece verifier on the learned graph (verifier split / merge 0.385 / 0.428 -> 0.437 / 0.504, partial); per-ray survival not tried.
- verified: checked

### CARD VS13 - HAWP: holistic attraction field proposals + the same LoI verification
- papers: Holistically-attracted wireframe parsing (Xue et al., CVPR 2020) https://arxiv.org/abs/2003.01663
- depth: M
- stage: S3, S4
- idea: Instead of enumerating all junction pairs, a 4-D attraction field proposes lines directly (about 4k per image against 22k for L-CNN); the same LoI head verifies them; a line survives only if both endpoints match junction proposals.
- formulation: AFM per pixel (d, theta, theta1, theta2), d normalised by d_max = 5; up to 3 candidates per point d' = d + kappa * delta_d, kappa in {-1, 0, 1}; junction head as in L-CNN (lambda_msk 8, lambda_off 0.25). Verification: s = 32 uniform points, bilinear (RoIAlign-like), 1-D max-pool, small FC head, BCE; positive if the endpoint-matched maximum distance to a GT line < eta = 1.5 (GT lines are added as positives), 300 positives + 300 negatives per image; refinement: both endpoints within tau = 10 of junction proposals. L = L_LS (channel-wise L1) + L_Junc + L_Ver. Stacked hourglass (2 stacks), 512 x 512, Adam 4e-4, 30 epochs, batch 6, one V100.
- input: feature map. output: scored segments.
- training: matching_free (thresholds).
- differentiable: partial (junction NMS and matching).
- sensors_data: images.
- results: Wireframe sAP10 / sAP15 / msAP / APH / FH: HAWP 66.5 / 68.2 / 65.7 / 84.5 / 80.3 vs re-trained L-CNN 63.6 / 65.3 / 62.9 / 81.6 / 77.9; YorkUrban msAP 28.1 vs 26.8; 29.5 FPS vs 15.6 FPS. Ablation: HAWP is less sensitive to negative samplers because the proposals are informative; no ablation of proposal count or K.
- compute: 29.5 FPS on one GPU.
- code: URL from memory, not opened: https://github.com/cherubicXN/hawp
- fit_to_us: reuse_idea - S - informative proposals ease the verifier = the teardrop's direction prior before verification; head unchanged from VS12.
- compat_needs: dense field head for proposals. compat_conflicts: as VS12.
- tried_before: none (see VS12 for E116).
- verified: checked

### CARD VS14 - DeepLSD: distance and angle fields with per-sample inlier verification
- papers: DeepLSD (Pautrat et al., CVPR 2023) https://arxiv.org/abs/2212.07766
- depth: M
- stage: S3, S5
- idea: The network predicts a line distance field (DF) and angle field (AF); a classical detector runs on a surrogate gradient built from them; each candidate line is verified by the fraction of 50 samples that agree with DF and AF; accepted lines are refined by least squares on 10 samples.
- formulation: DF = r e^{-D_n}, r = 5 px; surrogate gradient magnitude r - D, angle AF - pi/2; filter: n_f = 50 points per line, keep the line if more than 50 % of samples have DF < eta_DF = 1.5 and angle within eta_theta = pi/9; refinement: n_opt = 10 samples, cost = angle + mean DF + vanishing-point term, weights 1 / 1 / 0.2, only orientation and midpoint offset change; loss L1 on normalised DF + circular L2 on AF within r of a line; pseudo ground truth from homography adaptation (N not stated); Adam 1e-3, about 12 h on one RTX 2080.
- input: image. output: DF, AF, segments.
- training: self_supervised pseudo labels.
- differentiable: partial (verification and refinement are classical, after the network).
- sensors_data: images (Wireframe, MegaDepth).
- results: HPatches structural repeatability 0.367, orthogonal localisation error 0.818; York Urban vanishing-point error 1.63, AUC 85.6; refinement lowers HAWP / TP-LSD orthogonal error by up to 32 % (paper).
- compute: about 12 h training.
- code: URL from memory, not opened: https://github.com/cvg/DeepLSD
- fit_to_us: reuse_idea - S - free, non-learned verifier for a ray: fraction of samples where heat map and direction channel agree with the ray; a training-free baseline and sanity check for nu.
- compat_needs: dense distance / angle fields (our dense head: heat map + 2 direction channels).
- compat_conflicts: threshold on the inlier fraction is a hard decision.
- tried_before: none.
- verified: checked

### CARD VS15 - ScaleLSD: line verification without a neural verifier (support degree)
- papers: ScaleLSD (Ke, Tan, Zheng, Shen, Wu, Xue, arXiv 2025) https://arxiv.org/abs/2506.09369
- depth: M
- stage: S3
- idea: Keeps the HAWP attraction-field parametrisation but drops LoI pooling: a candidate line is scored by its support degree (number of pixels whose decoded junction pair matches it); training is self-supervised at scale.
- formulation: proposals with fewer than tau_l = 10 supporting pixels are discarded in training and pseudo-labelling, tau_l = 5 at inference; the paper states there are no neural modules for verification; BCE only for endpoint regression; pseudo labels from a synthetic bootstrap model (8 primitives) and an "LSD-Rectifier" that replaces the direction field by the classical level-line orientation; data about 16k synthetic, 20k Wireframe, over 10M images; 128 samples per line only for a 3-D reconstruction evaluation.
- input: image. output: scored segments.
- training: self_supervised.
- differentiable: partial.
- sensors_data: images.
- results: zero-shot repeatability HPatches Rep-5 (S) 0.367 vs 0.322 (HAWPv3), RDNIM 0.337 vs 0.278; vanishing-point error 1.55 vs 1.63 (DeepLSD); sAP and inference speed not stated.
- compute: pseudo-label generation 0.607 s per batch with the rectifier vs 3.653 s with 10-iteration homography adaptation.
- code: URL from memory, not opened: https://github.com/ant-research/scalelsd
- fit_to_us: reuse_idea - S - evidence that agreement with dense fields can replace a learned verifier.
- compat_needs: dense fields. compat_conflicts: none.
- tried_before: none.
- verified: checked

### CARD VS16 - LETR: line segments as transformer queries (no sampling along the line)
- papers: Line segment detection using transformers without edges (Xu et al., CVPR 2021) https://arxiv.org/abs/2101.01909
- depth: M
- stage: S4, S5
- idea: 1000 line-entity queries regress two endpoints and a confidence; a coarse (C5, stride 32) then a fine (C4, stride 16) encoder-decoder refines them; nothing is sampled along the line.
- formulation: each stage 6 encoder + 6 decoder layers, 256 channels, 8 heads; the fine decoder inherits the entities of the coarse one; Hungarian matching on lambda1 d(L^, L) - lambda2 p (lambdas not stated); focal-style adaptive BCE, L1 on endpoints (GIoU removed), deep supervision on all decoder layers with shared heads, background class weight reduced 10x; coarse 500 epochs, fine 325 (incl. 25 focal fine-tune), AdamW 1e-4, initialised from COCO DETR.
- input: multi-level image features. output: 1000 scored segments.
- training: hungarian_set.
- differentiable: partial (hungarian).
- sensors_data: images.
- results: Wireframe sAP10 65.2, sAP15 67.7, APH 86.3, FH 83.3 (ResNet-101); YorkUrban sAP10 29.4, APH 62.7, FH 66.9 (ResNet-50); 5.04 FPS on a V100.
- compute: parameters not stated.
- code: URL from memory, not opened: https://github.com/mlpc-ucsd/LETR
- fit_to_us: not_suitable - Hungarian matching and 1000 queries; only the coarse-to-fine inheritance is an idea.
- compat_needs: Hungarian. compat_conflicts: matching-free rule.
- tried_before: E31 / E32 matched sparse decoder dropped (grid +0.058 F1).
- verified: checked

### CARD VS17 - ELSD: midpoint heat map + angle / length, scoring without any sampling along the line
- papers: ELSD (Zhang, Luo, Qin, He, Liu, arXiv 2021) https://arxiv.org/abs/2104.14205
- depth: M
- stage: S3, S2
- idea: A segment is (midpoint, angle, length); a Gaussian-peak midpoint heat map multiplied by a "line-centerness" gives the only confidence; there is no verification or classifier; a separate descriptor branch pools features along the segment with a rotated RoI.
- formulation: endpoints = centre +- (rho / 2)(cos theta, sin theta); line-centerness = sqrt(min(d1, d2) / max(d1, d2)); final centre confidence = P_mid * P_centerness^0.5 (non-centerness suppression); fine sub-pixel offset; shared features 128 x 128, heads at 256 x 256. Losses: lambda_mid 25 (CornerNet-style focal, alpha 2, beta 4), lambda_cen 10 (weighted BCE), lambda_geo 1 (angle L1 x 300, length smooth-L1 x 10), lambda_off 3; descriptor: rotated RoI bilinear sample (width and number of sample points not stated), 1 x W' depth-wise conv, stride-S max-pool, FC, L2 norm, descriptor length 256 / 64 / 36, hard-negative triplet loss; Adam, 170 epochs, lr 1e-3, batch 16.
- input: image feature maps. output: segments + descriptors.
- training: dense heat-map targets, homography self-supervision for descriptors; matching_free.
- differentiable: yes except peak picking.
- sensors_data: images (Wireframe, YorkUrban).
- results: Wireframe sAP5 / 10 / 15 64.3 / 68.9 / 70.9, APH 87.2, FH 82.3, 42.6 FPS (Res34, 512); Lite (256) 57.4 / 63.1 / 65.5 at 107.5 FPS; YorkUrban sAP10 30.2; descriptor line matching F-score 74.7 vs 66.4 for LBD (Table 3).
- compute: 38 FPS with the descriptor branch.
- code: not checked.
- fit_to_us: reuse_idea - S - a symmetric "centerness" ratio (min / max of the two end distances) as a cheap centred-ness prior; confirms that later line detectors (LETR, ELSD, ScaleLSD) dropped sampled verification once proposals are informative.
- compat_needs: none. compat_conflicts: none.
- tried_before: none.
- verified: checked

### CARD VS18 - Cascade R-CNN: stages with increasing IoU thresholds
- papers: Cascade R-CNN (Cai, Vasconcelos, CVPR 2018) https://arxiv.org/abs/1712.00726
- depth: M
- stage: S5, L
- idea: A sequence of detectors; stage t is trained on the boxes output by stage t-1 with a higher IoU threshold for positives. Regression raises the IoU of the sample set, so the number of positives stays about constant while the threshold rises; the same cascade runs at test time (no train / test mismatch).
- formulation: thresholds U = {0.5, 0.6, 0.7} (3 stages after the RPN); b^t = f_{t-1}(x^{t-1}, b^{t-1}); features re-pooled at the new boxes; per-stage loss CE + lambda [y >= 1] L_loc, lambda = 1; regression targets normalised per stage (mean / variance values not stated); class-agnostic regression; final score = ensemble of the three stage classifiers; detach between stages not stated. Iterative bbox (one regressor repeated) degrades boxes above IoU 0.85 and gives no gain beyond two applications; integral loss (several classifiers on one distribution) leaves high-IoU classifiers short of positives.
- input: point_set of proposals + feature map. output: refined boxes + scores.
- training: IoU-threshold assignment per stage; matching_free.
- differentiable: partial (sample sets are re-selected by IoU between stages).
- sensors_data: images (COCO).
- results: COCO val ablation baseline: FPN+ 34.9 AP -> iterative bbox 35.4 -> integral loss 35.4 -> cascade 38.9; AP90 7.1 -> 8.1 / 6.9 / 15.8; test-dev ResNet-101 FPN+: 38.8 -> 42.8 AP, AP75 41.9 -> 46.3. (Text says ResNet-50 for the ablations but Table 6 gives 36.5 for FPN+ R50, not explained.)
- compute: FPN+ R50 parameters 165 M -> 272 M; test 0.095 -> 0.115 s per image (Titan Xp).
- code: URL from memory, not opened: https://github.com/zhaoweicai/cascade-rcnn
- fit_to_us: reuse_idea - S - tightening the positive radius per stage (spec: 1.0 / 0.5 / 0.3 m) is the direct analogue; key evidence is "same distribution at train and test".
- compat_needs: a hypothesis set per stage with positives defined by a geometric score (IoU -> point-to-polyline distance).
- compat_conflicts: IoU resampling between stages is discrete; in a fixed-shape tree it becomes re-labelling with a tighter radius.
- tried_before: E39 cascade refiner (+0.001 F1 / +0.014 route F, small, kept); E120 position refinement (+0.012 F1@1).
- verified: checked

### CARD VS19 - DINO: look-forward-twice, mixed query selection, contrastive denoising
- papers: DINO (Zhang et al., ICLR 2023) https://arxiv.org/abs/2203.03605
- depth: M
- stage: S5, S4, T
- idea: LOOK FORWARD TWICE: the box update of layer i is supervised through the NEXT layer's update applied to the undetached previous box, so the loss of layer i+1 also trains layer i. MIXED QUERY SELECTION: top-K encoder features initialise only the anchors; content queries stay learnable. CONTRASTIVE DENOISING: per GT box one positive query (small noise, reconstruct) and one negative (larger noise, predict no object).
- formulation: Delta b_i = Layer_i(b_{i-1}); b_i' = Update(b_{i-1}, Delta b_i); b_i = Detach(b_i') is what is passed forward; the prediction of layer i is b_i^pred = Update(b_{i-1}', Delta b_i) with the UNDETACHED b_{i-1}' (Deformable DETR detaches in between). CDN: noise scales lambda1 < lambda2 (appendix lambda1 = 1.0, lambda2 = 2.0; the hyper-parameter table lists box noise scale 0.4, not reconciled); a group has 2n queries for n GT boxes; groups set dynamically (about 100 CDN pairs, 1000 DN queries for the large model); L1 + GIoU boxes, focal classes incl. negatives. 900 queries (300 x 3 patterns), 6 + 6 layers, mixed-query K not stated.
- input: queries (point_set / boxes) + multi-scale rasters. output: boxes + classes.
- training: hungarian_set + denoising.
- differentiable: partial (hungarian).
- sensors_data: images (COCO).
- results: COCO val ResNet-50, 12 epochs, AP: DN-DETR 43.4 -> optimised 44.9 -> + pure query selection 46.5 -> + mixed query selection 47.0 -> + look forward twice 47.4 -> + contrastive denoising 47.9; full DINO-4scale 49.0 (12 ep), 50.4 (24 ep), 50.9 (36 ep).
- compute: not stated here.
- code: URL from memory, not opened: https://github.com/IDEA-Research/DINO
- fit_to_us: reuse_idea - S - the look-forward-twice residual (undetached previous position + later delta in the loss) gives "no detach" credit assignment without sending gradients through the sampling positions; denoising is the DETR-side analogue of noisy teacher forcing.
- compat_needs: iterative per-stage position update.
- compat_conflicts: Hungarian matching for the main queries.
- tried_before: E39 cascade refiner small gain; E31 / E32 matched sparse decoder dropped.
- verified: checked

### CARD VS20 - Deformable DETR: reference points, learned offsets, multi-scale, iterative box refinement (+ code facts)
- papers: Deformable DETR (Zhu et al., ICLR 2021) https://arxiv.org/abs/2010.04159 ; code files read https://github.com/fundamentalvision/Deformable-DETR
- depth: M+C
- stage: S2, S5
- idea: Each query reads only K points around a reference point (position = reference + learned offset) with attention weights from the query itself; cost does not depend on map size. The decoder refines the box layer by layer.
- formulation: DeformAttn(z_q, p_q, x) = sum_m W_m [sum_k A_mqk W'_m x(p_q + dp_mqk)]; one linear layer of 3MK channels: the first 2MK are offsets (unconstrained range), the last MK a softmax over k (multi-scale: over l and k jointly); multi-scale M = 8, K = 4, L = 4 (C3 - C5 + C6). INIT (appendix + code): offset-layer weights 0, bias = 8 fixed directions (cos / sin of the head angle, scaled so the larger component is +-1) times (k + 1) for the k-th point; attention layer weights and bias 0 (uniform weights); the paper says decoder offset biases are further multiplied by 1/(2K). FORWARD (code): location = reference + offset / [W_l, H_l], i.e. offsets are in pixels of each level; for 4-D reference boxes offset / n_points * (w, h) * 0.5; no clamp, tanh or bound anywhere in the file. OPTIMISER (`main.py`): parameters named sampling_offsets and reference_points learn at lr x 0.1 (2e-4 -> 2e-5), backbone 2e-5, gradient clip 0.1. Decoder: reference = linear + sigmoid of the query; iterative refinement b^d = sigmoid(db^d + sigmoid^-1(b^{d-1})) with the gradient blocked at the previous box ("stabilisation", cites RAFT); two-stage: each encoder pixel proposes a box (scale 0.05), top boxes (no NMS) become decoder boxes. Defaults: 300 queries, 6 + 6 layers, 4 levels, 4 + 4 points; costs 2 / 5 / 2 (class / L1 / GIoU). `ms_deform_attn_core_pytorch` is a grid_sample reference (locations 2 * loc - 1, bilinear, zero padding, align_corners False; comment "for debug and test only"), but the module imports the compiled `MultiScaleDeformableAttention` at top level, so the function must be copied out to run without the CUDA build.
- input: bev_raster / image_feat pyramids + queries with reference points. output: boxes + classes.
- training: hungarian_set, focal + L1 + GIoU, auxiliary losses on every decoder layer.
- differentiable: partial (hungarian; detached reference).
- sensors_data: images (COCO).
- results: COCO AP / FPS: DETR 42.0 (500 ep) 28 FPS; Deformable DETR 43.8 (50 ep) 19 FPS; + iterative refinement 45.4; + two-stage 46.2; test-dev 52.3 with ResNeXt-101 + DCN + TTA.
- compute: O(2 N_q C^2 + min(HW C^2, N_q K C^2)); 1.6x faster than DETR-DC5, about 25 % slower than Faster R-CNN + FPN.
- code: checked (three files above).
- fit_to_us: reuse_idea / reimplement - S - 4 levels x 4 points around a tip is one grid_sample call; the init (a fixed fan of directions scaled by k, zero weights) is the precedent for "hand-built teardrop + zero-initialised learned offsets"; lr x 0.1 for offsets is a cheap stabiliser.
- compat_needs: Hungarian for the decoder as published; queries with reference points.
- compat_conflicts: Hungarian (matching-free rule); unbounded offsets can leave the BEV range (zero padding then reads "no lane").
- tried_before: E31 / E32 matched sparse decoder dropped; E17 / E18 transformer decoder -0.11.
- verified: checked

### CARD VS21 - RAFT: pyramid-window lookup around the current estimate + recurrent residual update (+ code facts)
- papers: RAFT (Teed, Deng, ECCV 2020) https://arxiv.org/abs/2003.12039 ; code files read https://github.com/princeton-vl/RAFT
- depth: M+C
- stage: S2, S5
- idea: Keep one current estimate (flow); each iteration look up a local window of the correlation volume around the CURRENT estimate on every pyramid level, feed it to a ConvGRU, and add a residual. Weights are shared across iterations.
- formulation: correlation volume = dot products of 1/8-resolution features (256 ch), 4-level pyramid by 2x2 average pooling (kernels 1, 2, 4, 8 in the paper); lookup radius r = 4, the paper defines the neighbourhood as an L1 ball but the code uses a SQUARE (2r + 1)^2 = 81-offset window per level, offsets applied in each level's own coordinates (centroid / 2^i), bilinear sampler, 4 x 81 = 324 channels; ConvGRU (separable 1x5 / 5x1) with correlation, flow and context inputs; f_{k+1} = f_k + Delta f; sequence loss sum_i gamma^{N-i} |f_gt - f_i|_1, gamma = 0.8; `coords1` is detached at the top of every iteration (gradient only through Delta f); 12 iterations in training (code default), 32 (Sintel) / 24 (KITTI) at test; hidden state from tanh of a context-network part, context input from relu of the rest; convex upsampling.
- input: two feature maps (rasters) + current estimate. output: flow field (dense).
- training: supervised sequence loss; matching_free.
- differentiable: yes with a stop-gradient on the estimate.
- sensors_data: images (Sintel, KITTI).
- results: Sintel train clean / final EPE 1.43 / 2.71; KITTI-15 train EPE 5.04, F1-all 17.4 %; inference updates (Sintel clean / final EPE): 1: 4.04 / 5.45, 3: 2.14 / 3.52, 8: 1.61 / 2.88, 32: 1.43 / 2.71, 100: 1.41 / 2.72, 200: 1.40 / 2.73 (no divergence).
- compute: 5.3 M parameters; 0.10 s with 10 updates on a GTX 1080 Ti.
- code: checked (`corr.py`, `raft.py`).
- fit_to_us: reuse_idea - S - a pyramid-window lookup around the current tip estimate (r = 4 cells on f0 / f1 / f2 = 2 / 4 / 8 m half-width at 0.5 / 1 / 2 m cells; 3 x 81 = 243 reads per query) is a drop-in alternative to rings; weight-shared residual update with a gamma-weighted sequence loss is the matching-free multi-step loss.
- compat_needs: dense pyramid with bilinear lookup (we have it).
- compat_conflicts: a window lookup is isotropic; the teardrop is anisotropic by design.
- tried_before: E94 fan of 27 bilinear samples + 7 cross-section points (a fixed-pattern lookup); E120 refinement (+0.012 F1@1).
- verified: checked

### CARD VS22 - Sparse R-CNN: learned proposals, instance-conditioned dynamic head, 6 stages with blocked box gradient
- papers: Sparse R-CNN (Sun et al., CVPR 2021) https://arxiv.org/abs/2011.12450
- depth: M
- stage: S5, S2
- idea: N learnable proposal boxes plus learnable proposal features replace the RPN; each box pools RoI features and a dynamic head whose two 1x1 convolutions are generated from the proposal feature transforms them; boxes and features are refined over 6 stages.
- formulation: N = 100 (300 / 500 in variants), d = 256; RoIAlign S x S bins (7 x 7 not stated); self-attention among object features, then dynamic conv (C -> C/4 -> C, ReLU, norm) with parameters from the proposal feature; 3-layer MLP box, linear class; stage t+1 takes refined boxes and features of stage t; gradient blocked at the proposal boxes of every stage except the initial ones; loss 2 focal + 5 L1 + 2 GIoU on matched pairs (optimal bipartite matching).
- input: pyramid + queries. output: boxes + classes.
- training: hungarian_set.
- differentiable: partial (matching; box gradient blocked).
- sensors_data: images (COCO).
- results: COCO val ResNet-50 FPN 36 epochs: 42.8 AP at 23 FPS (45.0 with 300 boxes); stages AP / FPS: 1: 21.7 / 35, 2: 36.2 / 33, 3: 39.9 / 29, 6: 42.3 / 23, 12: 41.6 / 17; box initialisation centre 41.5, image 42.3, grid 41.0, random 42.1 ("relatively robust").
- compute: V100 FPS above.
- code: URL from memory, not opened: https://github.com/PeizeSun/SparseR-CNN
- fit_to_us: reuse_idea - S - evidence that a few stages saturate and the starting hypotheses hardly matter.
- compat_needs: Hungarian. compat_conflicts: matching-free rule.
- tried_before: E31 / E32 matched sparse decoder dropped.
- verified: checked

### CARD VS23 - DAB-DETR: queries as dynamic 4-D anchors with width / height-modulated attention
- papers: DAB-DETR (Liu et al., ICLR 2022) https://arxiv.org/abs/2201.12329
- depth: M
- stage: S2, S5
- idea: Each query IS an anchor (x, y, w, h) updated layer by layer; the positional attention is modulated by the ratio of a reference width / height (predicted from the content) to the anchor size, giving an anisotropic, size-aware window.
- formulation: P_q = MLP(PE(A_q)), sinusoidal PE of all four coordinates, temperature T = 20 instead of 10000 (coordinates in [0,1]); cross-attention query concatenates content and PE(x_q, y_q) * MLP(C_q); modulated attention = (PE(x) PE(x_ref) w_ref / w + PE(y) PE(y_ref) h_ref / h) / sqrt(D), (w_ref, h_ref) = sigmoid(MLP(C_q)); shared head predicts (dx, dy, dw, dh) per layer (update space and detach not stated on the page seen); 300 queries.
- input: queries (anchors) + feature map. output: boxes.
- training: hungarian_set.
- differentiable: partial.
- sensors_data: images (COCO).
- results: R50 50 epochs AP: 42.2 (42.6 with 3 patterns), DC5 44.5 (45.7); DC5 ablation: no anchor update 44.0 (update adds 1.7), no w/h modulation 45.0, 2-D anchor points instead of 4-D boxes 44.0, no temperature tuning 44.4, full 45.7.
- compute: not stated.
- code: URL from memory, not opened: https://github.com/IDEA-Research/DAB-DETR
- fit_to_us: reuse_idea - S - a per-tip (length, width) gain that modulates the sampling window (spec: gain in [0.5, 1.5]); 4-D anchors beat 2-D points by 1.7 AP in their ablation.
- compat_needs: anchor with size. compat_conflicts: Hungarian.
- tried_before: none.
- verified: checked

### CARD VS24 - Sparse4D v1 / v3: fixed + learnable keypoints scaled by the instance box, bounded offsets, quality head
- papers: Sparse4D (Lin et al., arXiv 2022) https://arxiv.org/abs/2211.10581 ; Sparse4D v3 (Lin et al., arXiv 2023) https://arxiv.org/abs/2311.11722
- depth: M
- stage: S2, S3, S8
- idea: Each instance (anchor box + feature) samples a fixed keypoint set (box centre, face centres) plus learnable keypoints whose offsets are bounded and expressed in box units; v3 adds a quality head (centerness, yawness) whose product with the confidence ranks detections.
- formulation: v1: 7 fixed + 6 learnable keypoints; D_m = R_yaw [sigmoid(Phi(F_m)) - 0.5], so every learnable offset lies in [-0.5, 0.5] box units; P_L = D_m * [w, h, l] + [x, y, z]; temporal keypoints shifted with constant velocity and ego motion; bilinear reads per view n, scale s, time t; weights W = Psi(F_m) (linear layer, groups G, value not stated); fusion order views / scales, then time, then keypoints. v3: centerness C = exp(-|pos_pred - pos_gt|_2), yawness = sin / cos product, loss lambda1 CE(Y) + lambda2 focal(C) (lambdas not stated), score = centerness x confidence; 5 denoising groups (3 temporal); decoupled attention (concatenate instead of add); 900 instances, 600 temporal, 6 layers.
- input: image_feat pyramids + instance anchors. output: 3-D boxes, tracking ids.
- training: bipartite matching + denoising (Hungarian family).
- differentiable: partial.
- sensors_data: nuScenes cameras.
- results: v3 ablation nuScenes val R50 (mAP / NDS): v2 0.439 / 0.539; + single-frame denoising 0.447 / 0.548; + decoupled attention 0.458 / 0.551; + temporal denoising 0.462 / 0.557; + centerness 0.463 / 0.554; + yawness 0.469 / 0.561; v1: learnable keypoints on vs off 0.537 vs 0.533 NDS (the text's percentage gains do not match this table).
- compute: v3 R50 256 x 704 19.8 FPS.
- code: URL from memory, not opened: https://github.com/linxuewu/Sparse4D
- fit_to_us: reuse_idea - S - cleanest published recipe for "teardrop = reference pattern scaled by (length, width) + bounded learned offsets"; the quality head is a calibrated nu for the beam.
- compat_needs: per-tip size. compat_conflicts: the perspective projection is not needed.
- tried_before: none.
- verified: checked

### CARD VS25 - DAT: deformable attention with tanh-bounded offsets
- papers: Vision transformer with deformable attention (Xia et al., CVPR 2022) https://arxiv.org/abs/2201.00520
- depth: M
- stage: S2
- idea: Reference points form a uniform grid; a small network predicts shifts; the shifts are bounded by a scaled tanh to keep sampling local.
- formulation: offset network = 5x5 depthwise conv, GELU, 1x1 conv without bias -> 2 channels per group; Delta p <- s * tanh(Delta p), s = 2 ("without additional tuning"); G offset groups (3 / 6 for DAT-T and DAT-S stages 3 / 4); bilinear sampling; relative position bias interpolated at continuous displacements. No ablation with versus without the bound; a sweep of s from 0 to 16 is stable (values only in a figure). Learning-rate scaling of the offset net: not stated.
- input: feature map. output: sampled keys / values.
- training: image classification; none matching.
- differentiable: yes.
- sensors_data: ImageNet-1K.
- results: DAT-T top-1: no offsets, no position embedding 81.4; offsets only 81.8; relative position bias only 81.7; both 82.0.
- compute: not stated.
- code: URL from memory, not opened: https://github.com/LeapLabTHU/DAT
- fit_to_us: reuse_idea - S - offset = r_max * tanh(.) with r_max a fraction of the ring gap.
- compat_needs: none. compat_conflicts: none.
- tried_before: none.
- verified: checked

### CARD VS26 - CLRNet: lane priors refined over three pyramid levels, ROIGather, Line-IoU (+ code facts)
- papers: CLRNet (Zheng et al., CVPR 2022) https://arxiv.org/abs/2203.10350 ; code file read https://github.com/Turoad/CLRNet
- depth: M+C
- stage: S2, S5, L
- idea: Lane priors (start point, angle, length, N = 72 offsets) start uniformly over the image and are refined in three stages that read feature levels from high-level (semantic) to low-level (detailed); each stage pools N_p = 36 points along the current prior and adds a global attention context; a Line-IoU loss scores the whole lane.
- formulation: P_t = P_{t-1} o R_t(L_{t-1}, P_{t-1}), T = 3 over FPN levels L0..L2 (strides not stated; the code takes the last three feature maps reversed, stage 0 = deepest); learnable priors (as Sparse R-CNN); ROI features: bilinear samples along the prior, resized map 10 x 25 with C = 64, earlier-level ROI features concatenated, conv + FC to a C x 1 vector per prior; ROIGather W = softmax(X_p^T X_f / sqrt(C)), G = W X_f^T added back; two FC layers give the refinement. Line-IoU: each x_i extended by radius e = 15 px, d_O = min(x_p + e, x_g + e) - max(x_p - e, x_g - e), d_U = max(...) - min(...), LIoU = sum d_O / sum d_U in [-1, 1], loss 1 - LIoU. Assignment: dynamic top-k (YOLOX-style), cost w_sim C_sim + w_cls C_cls, C_sim = (C_dis C_xy C_theta)^2, w_cls = 1, w_sim = 3; top-1 gives the NMS-free variant. Loss = focal (cls) + smooth-L1 (start, angle, length) + LIoU (weights 2 / 4 / 6 tried, 2 best). CODE: 192 priors, 36 sampled points, 64 channels; priors are DETACHED before pooling in the next stage (geometric lines only, offset residuals not carried); EVERY stage gets classification, regression and IoU losses (averaged over stages); defaults cls 2.0, xyt 0.5, IoU 2.0, segmentation 1.0 (training only); grid_sample bilinear, align_corners True, zero padding outside the image; points after the predicted end are invalidated at inference. Input 320 x 800, AdamW 1e-3, 15 epochs (CULane).
- input: image_feat / bev_raster levels + point_set of priors. output: polyline (lane) with length.
- training: dynamic top-k assignment (matching_free in our sense), no Hungarian.
- differentiable: partial (top-k assignment, detached priors).
- sensors_data: images (CULane, TuSimple, LLAMAS).
- results: CULane F1@50 / mF1 / F1@75: R18 79.58 / 55.23 / 62.21 (119 FPS), DLA34 80.47 / 55.64 / 62.78 (94 FPS); ablation mF1: baseline 51.90, + LIoU 52.80, + refinement 54.74, + ROIGather 55.23; refinement order R0 -> R1 -> R2 54.74 vs reverse 53.58 vs single stage 52.80; smooth-L1 54.22 vs LIoU 55.23.
- compute: 11.9 GFLOPs (R18).
- code: checked.
- fit_to_us: reuse_idea / reimplement - S / M - (a) pool N_p points along the current polyline every stage; (b) coarse-to-fine, deepest level first, beat the reverse by 1.2 mF1 - supports f2 first, f0 last; (c) LineIoU-style overlap is a matching-free whole-lane loss (polyline version: overlap of buffered segments); (d) the code is the clearest instance of "detach the geometry, supervise every stage".
- compat_needs: priors as lines with a start point; any raster features.
- compat_conflicts: row-parametrised offsets (one x per row) are invalid for BEV graphs of arbitrary heading.
- tried_before: E39 cascade refiner small gain; E94 reads at fixed bearings.
- verified: checked

### CARD VS27 - LaneATT: anchor-pooled features + anchor-based attention
- papers: Keep your eyes on the lane (Tabelini et al., CVPR 2021) https://arxiv.org/abs/2010.12035
- depth: M
- stage: S2, S4
- idea: Fixed line anchors (origin on the image border + angle) pool the feature column the anchor passes through at each row; an attention layer lets each anchor aggregate the features of the other anchors.
- formulation: N_pts = 72 points per lane at fixed y; 2,782 anchors in total, top N_anc = 1000 kept by a training-based filter (angles and origins not stated); backbone feature reduced by a 1x1 conv to C_F channels; a_i^loc = feature at the anchor's x for each feature row (zero padding outside; C_F * H_F values); attention weights w_ij = softmax over j != i of an FC layer on a_i^loc; a_i^glob = sum_j w_ij a_j^loc; concatenate local and global; heads: class (K + 1 = 2) and (length + N_pts offsets); positive if mean x-distance < tau_p = 15, negative if > tau_n = 20; focal + smooth-L1; NMS with the same distance (thresholds not stated).
- input: image_feat + anchors. output: polyline with start and length.
- training: threshold assignment (matching_free).
- differentiable: yes except NMS.
- sensors_data: images (TuSimple, CULane, LLAMAS).
- results: CULane F1 R18 / R34 / R122 75.09 / 76.68 / 77.02 (250 / 171 / 26 FPS); ablation R34: without anchor-based pooling 64.89 (vs 76.68), without attention 75.78, without focal 75.54; anchors 250 / 500 / 750 / 1000 / 1250: F1 68.68 / 75.45 / 75.80 / 76.66 / 75.91.
- compute: 22.13 M parameters (R34), 18.0 GMACs.
- code: URL from memory, not opened: https://github.com/lucastabelini/LaneATT
- fit_to_us: reuse_idea - S - the largest ablation effect is line-aligned pooling (+11.8 F1); anchor count saturates near 1000 and degrades beyond.
- compat_needs: anchors as lines with fixed row sampling. compat_conflicts: one x per row; anchors from a border.
- tried_before: none.
- verified: checked

### CARD VS28 - Polar R-CNN: local polar poles propose anchors, global head, NMS-free O2O branch
- papers: Polar R-CNN: end-to-end lane detection with fewer anchors (arXiv 2024) https://arxiv.org/abs/2411.01499
- depth: M (all pages read, including the appendix)
- stage: S1, S2, S4
- idea: Anchors are lines in polar form (radius r, angle theta) relative to a pole; every lattice point of a coarse feature map ("local pole") predicts the (r, theta) of the nearest lane and a confidence, so only the top-K (about 20) anchors reach the second stage instead of about 192; the second stage samples features along each anchor; an O2O branch trained with stopped gradients removes duplicates without NMS.
- formulation: local polar module: highest FPN level downsampled, 1x1 conv regression (theta, r), classification of two 1x1 convs; pole positive if its radius < lambda_l (value not stated); labels r^ = distance pole -> lane, theta^ = direction to the nearest lane point; local map H x W = 4 x 10 (6 x 13 for CurveLanes); K = 20 (CULane, TuSimple, LLAMAS), 12 (DL-Rail), 50 (CurveLanes). Global polar module: one global pole near the static vanishing point; 36 sampled points per anchor (72 regression points), features of three FPN levels combined by a learned softmax-weighted sum, then a linear projection; GNN-style relation block (not attention). Triplet head: O2M cls + reg (SimOTA, dynamic k = 4), O2O cls (Hungarian) with adjacency A = A^C (confidence) x A^G (|dtheta| < tau_theta, |dr| < lambda_g), max-pool over neighbours + 3-layer MLP; dual selection s~ > tau_o2o and s > tau_o2m (CULane 0.46 / 0.48). Losses BCE + smooth-L1 (local), focal + rank + GIoU + smooth-L1 endpoints + auxiliary segment loss (global); the appendix redefines the lane IoU (GLaneIoU: per-point semi-width from the local direction, overlap / gap / union), g = 0 for assignment costs and g = 1 for the GIoU loss. 800 x 320, AdamW lr 0.006, cosine, one A100.
- input: image_feat levels. output: polyline lanes, no NMS.
- training: SimOTA (O2M) + hungarian_set (O2O); gradient stopped on the O2O branch.
- differentiable: partial.
- sensors_data: images (TuSimple, CULane, LLAMAS, CurveLanes, DL-Rail).
- results: CULane F1@50 / F1@75 R18 80.81 / 63.96 (NMS-free), DLA34 81.49 / 64.97; CurveLanes DLA34 F1@50 87.29; 20 proposed anchors vs 192 (CLRNet); ablation CULane F1@50: fixed anchors 79.90, fixed + aux loss 80.38, proposals only 75.85, full 80.81; stop-gradient on the O2O branch 71.68 -> 80.81; CurveLanes NMS threshold 50 px 85.38, best 15 px 86.81, NMS-free 87.29. FPS and parameters not stated numerically; no ablation of sampled-point number or K in the text.
- compute: one A100-40G for training.
- code: not found on the pages read (URL from memory, not opened: https://github.com/ShqWW/PolarRCNN)
- fit_to_us: reuse_idea - S / M - every grid point proposing its nearest lane in polar form (r, theta) is a possible tip / seed generator; the stop-gradient between the ranking branch and the main branch is a case where cutting gradients is essential (71.68 -> 80.81).
- compat_needs: Hungarian for O2O.
- compat_conflicts: matching-free rule for the O2O head; anchors are image-plane lines.
- tried_before: none directly (E94 seeds = heat-map peaks, 192).
- verified: checked

### CARD VS29 - BezierFormer: Bezier curve queries with several reference points along the curve, matching-free assignment
- papers: BezierFormer: a unified architecture for 2D and 3D lane detection (arXiv 2404.16304) https://arxiv.org/abs/2404.16304 ; related TopoCurve (poster abstract only) https://neurips.cc/virtual/2026/poster/149504
- depth: M (BezierFormer), A (TopoCurve)
- stage: S2, S5, L
- idea: A query is a cubic Bezier curve; deformable attention is applied at N_ref = 5 reference points spread along the curve, so the receptive field covers the whole lane; assignment is SimOTA with top-k = 1 (NMS-free, no Hungarian); the regression loss is a Chamfer IoU on sampled curve points.
- formulation: reference points at t = 0, 0.25, 0.5, 0.75, 1; N_sample = 5 sampling points per reference point per feature scale; two MLPs predict offsets and weights, offsets added to the reference points (bound not stated); attention = sum over reference points, scales and samples with weights normalised to 1; queries 10 (CULane), 16 (CurveLanes), 32 (OpenLane); loss: Chamfer IoU on 200 sampled points + length loss + endpoint loss, focal classification, lane width e = 10 (CULane, CurveLanes) or 0.9 (OpenLane); optional perspective projection of 3-D reference points onto the front view.
- input: image_feat levels + queries (bezier). output: bezier lanes.
- training: matching_free (SimOTA, top-k = 1).
- differentiable: partial (assignment).
- sensors_data: images (OpenLane, CULane, CurveLanes).
- results: OpenLane F1 / category acc: ResNet18 53.84 / 92.02, Swin-T 56.43 / 94.1; CULane F1 79.44 (R18), 80.63 (Swin-T); CurveLanes F1 89.06 (R18), 91.31 (Swin-T) (the text cites 90.72 / 91.06 elsewhere). Ablation CurveLanes / OpenLane F1: ordinary attention 82.76 / 46.53; deformable 85.81 / 49.08; Bezier attention one reference point (t = 0.5) 87.60 / 52.13; five reference points 89.06 / 53.84; representations: polynomial 80.90 / 50.30, row-based 83.10 / 51.27, Bezier 89.06 / 53.84; losses: sampling loss 78.78 / 47.11, sampling IoU 81.86 / 48.32, Chamfer IoU 89.06 / 53.84. TopoCurve poster: 50.6 OLS and DET_p 56.8 on OpenLane-V2 (abstract only).
- compute: not stated.
- code: not checked.
- fit_to_us: reuse_idea / reimplement - M - (a) the strongest evidence found for curve-aligned multi-point reads (CurveLanes / OpenLane F1: one Bezier reference point over plain deformable attention +1.8 / +3.1; five reference points over one +1.5 / +1.7; five over deformable +3.3 / +4.8); (b) Chamfer IoU is a matching-free curve loss close to our point-to-polyline loss; (c) SimOTA top-1 shows an NMS-free, Hungarian-free assignment for curve queries.
- compat_needs: curve queries with control points (bezier <-> polyline adapter).
- compat_conflicts: cubic Bezier cannot represent a branching ray.
- tried_before: E53 Bezier connectors (kept in system A); E120 point-to-polyline loss both ways is similar to a Chamfer loss.
- verified: checked

### CARD VS30 - BezierLaneNet: one cubic Bezier per lane, sampled-point L1, column proposals
- papers: Rethinking efficient lane detection via curve modeling (Feng et al., CVPR 2022) https://arxiv.org/abs/2203.02431
- depth: M
- stage: S2, S4, L
- idea: A lane is a cubic Bezier curve (4 control points, 8 numbers); the loss compares 100 points sampled at uniform t on the predicted and GT curve, so the curve is never discretised inside the network.
- formulation: Bernstein basis precomputed at fixed t, so sampling is one matrix multiplication; L1 over n = 100 samples; Hungarian matching with quality Q = score^(1-alpha) (1 - L1)^alpha, alpha = 0.8; weighted BCE (negative weight 0.4); total lambda1 = 1 (regression), lambda2 = 0.1 (classification), lambda3 = 0.75 (auxiliary segmentation); GT control points by least squares; feature flip fusion (feature map fused with its horizontal flip through a deformable conv); column proposals: average pooling along the vertical axis gives W/16 proposals (50 on CULane), two 1x3 1-D convs, then 1x1 classification and regression; layer-3 backbone features with dilated blocks (4, 8), 16x downsampling.
- input: image_feat. output: bezier + existence per column proposal.
- training: hungarian_set.
- differentiable: partial (matching).
- sensors_data: images (TuSimple, CULane, LLAMAS).
- results: LLAMAS test F1 95.17 (R34), 94.91 (R18); CULane F1 75.41 (R34 ablation row); 213 FPS (R18, 4.10 M parameters), 150 FPS (R34, 9.49 M).
- compute: above.
- code: URL from memory, not opened: https://github.com/voldemortX/pytorch-auto-drive
- fit_to_us: reuse_idea - S - Bernstein-matrix sampling is the cheapest differentiable curve resampler (adapter bezier <-> polyline); sampled-point L1 is a matching-free curve loss once correspondence is fixed by the parameter t.
- compat_needs: one curve per lane, Hungarian.
- compat_conflicts: a single cubic cannot cover a 24 m branch with a bend and a fork; column pooling presumes lanes run vertically.
- tried_before: E53 Bezier connectors.
- verified: checked

### CARD VS31 - LaneSegNet: lane attention (reference points along the segment, identical initialisation, heads-to-regions)
- papers: LaneSegNet (Li et al., ICLR 2024) https://arxiv.org/abs/2312.16108
- depth: M
- stage: S2, S5, S6
- idea: Deformable attention whose reference points lie uniformly along the predicted lane segment (4 per boundary = 8) and whose different heads attend different regions of the segment; in the first decoder layer all heads share the SAME reference point from the positional query (identical initialisation).
- formulation: LaneAttn(q_i, p_i, B) = sum_m W_m [sum_k a_imk W'_m Bilinear(B, p_i,m + dp_imk)]; offsets from a linear layer of the query, initialised so that K = 32 sampling locations lie evenly around each reference point (4 distances x 8 directions, as in Deformable DETR); single-scale BEV (200 x 100 cells over +-50 m x +-25 m = 0.5 m cells); 6 decoder layers (self-attention 8 heads, lane attention 8 heads, FFN 512); 200 queries of 512 dims (256 reference part + 256 instance part); lane segment = centreline + two boundaries, 10 points each; losses: L1 0.025, mask CE 3.0 + dice 1.0, focal class 1.5, line type 0.01, topology focal 5.0; Hungarian (or Sinkhorn); AdamW 2e-4, 24 epochs, 8 V100.
- input: bev_raster. output: polyline segments + adjacency.
- training: hungarian_set.
- differentiable: partial (matching).
- sensors_data: OpenLane-V2 camera BEV (the paper says nuScenes lacks lane-segment labels, so it is not used).
- results: ablation (mAP / TOP_lsls): deformable attention one reference point 28.7 / 6.9; hierarchical queries 29.1 / 5.2; multi-point attention 24.6 / 6.9; lane attention 32.6 / 8.1; without heads-to-regions and identical init 28.8 / 5.6; full model 32.6 mAP at 14.7 FPS (tiny 28.5 at 16.2 FPS); centreline reshaped DET_l 31.8 vs TopoNet 24.9; latency 20.67 ms vs 30.14 ms for a multi-branch baseline.
- compute: above.
- code: URL from memory, not opened: https://github.com/OpenDriveLab/LaneSegNet
- fit_to_us: reuse_idea / reimplement - M - closest published BEV-grid read: bilinear reads on a 0.5 m BEV raster around reference points on the hypothesis; adding reference points indiscriminately hurt (24.6), the head-to-region split is the gain.
- compat_needs: BEV raster (ours), Hungarian for the decoder as published.
- compat_conflicts: Hungarian; mmcv / mmdet3d codebase (not in our env).
- tried_before: E31 / E32 dropped; E17 / E18 transformer decoder -0.11.
- verified: checked

### CARD VS32 - TopoBDA: Bezier deformable attention (control points as reference points) with camera + radar
- papers: TopoBDA (Kalfaoglu et al., arXiv 2024; Neurocomputing 2026 per the earlier review) https://arxiv.org/abs/2412.18951
- depth: M (about 90 % of the page read)
- stage: S2, S5, B
- idea: Each Bezier control point is its own attention head and reference point, with K learned offsets per point; the curve's shape steers where the BEV is read, without resampling the curve.
- formulation: BDA(q, V, C) = sum_{n=0}^{N} sum_{k=1}^{K} A_nk W_n V(c_n + dp_nk); 4 control points (the conclusion mentions 8; supplement text says 4); K = 32 attention offsets per control point; offset bounds not stated; control points of the first layer from an MLP + sigmoid, later layers add offsets in inverse-sigmoid space; decoder BDA then self-attention, 10 layers in the runtime study; auxiliary instance masks (training only); one-to-many auxiliary decoder sharing weights (repetitions R: 0 -> OLS_l 46.5, 4 -> 48.0, 5 -> 48.1); Hungarian with a mask + L1 mixed cost (Mask DINO style); weights lambda_reg 3 (matcher 5), mask BCE 5, Dice 5, class 2; BEV 200 x 104 cells of 0.5 m, 20 height bins of 1 m in [-10, 10], multi-height Lift-Splat; radar points (x, y, z + C_radar features) assigned to the nearest voxel, concatenated with camera features, flattened over height, 2-D conv; SD map concatenated in BEV; sweeps / radar representation not stated.
- input: bev_raster + queries (bezier). output: bezier centrelines (+ topology MLP head).
- training: hungarian_set + one-to-many.
- differentiable: partial.
- sensors_data: OpenLane-V2 subsets A (Argoverse 2) and B (nuScenes): camera, + radar, + lidar, + SD map.
- results: attention ablation SwinB OLS_l: SA 41.0, MA 42.6, SPDA 44.1, MPDA4 47.4, MPDA16 47.5, BDA 48.0 (BDA +3.9 over SPDA, +0.5 over MPDA16); subset B ResNet50 OLS_l: camera 49.9, + radar 54.8 (+4.9), + lidar 61.6; subset A: camera 44.1, + SD map 50.1; decoder runtime (10 layers, torch) BDA 18.47 ms, MA 24.76, SA 15.96; cost per BDA layer 10.0 GFLOPs, 1.48 GB, 14.78 M parameters vs SA 28.4 GFLOPs, 1.995 GB, 16.04 M and MA 41.6 GFLOPs, 5.07 GB, 16.24 M (Table S3); Table 5 columns could not be mapped on the extraction.
- compute: above.
- code: not checked.
- fit_to_us: reuse_idea - S - the only published camera + RADAR lane-topology result (radar +4.9 OLS_l on nuScenes subset B); BDA is the cheapest "read at hypothesis control points" pattern (K = 32 per control point, on a 0.5 m BEV).
- compat_needs: BEV raster + Bezier queries.
- compat_conflicts: Hungarian.
- tried_before: E53 Bezier connectors; E31 / E32.
- verified: checked

### CARD VS33 - CondLaneNet: start-point proposals, conditional convolution, row-wise shape and per-row range
- papers: CondLaneNet (Liu et al., ICCV 2021) https://arxiv.org/abs/2105.05003
- depth: M
- stage: S1, S2, S3
- idea: Each lane instance is detected at its START point (CenterNet-style heat map); a dynamic kernel generated at that point convolves the feature map into that lane's row-wise shape; the vertical range is a per-row binary "does the lane pass this row"; a recurrent module lets forks share one start point.
- formulation: proposal heat map 1 x Hp x Wp + parameter map Cp x Hp x Wp (sizes not stated); x per row = expectation of a softmax over X grid cells (loc = sum_j j p_ij) plus an offset map; range: per-row logistic output, lane spans v_min..v_max; RIM = LSTM emitting at each step a 2-way state (continue / stop) and a kernel vector, so one proposal yields several lanes until "stop"; loss = L_point (focal) + 1.0 L_row (L1 on expected x) + 1.0 L_range (BCE per row) + 0.4 L_offset + 1.0 L_state (CE); RIM only on CurveLanes.
- input: image_feat. output: polyline lane instances (start point, rows, range).
- training: dense, no Hungarian.
- differentiable: yes except peak picking.
- sensors_data: images (CULane, TuSimple, CurveLanes).
- results: F1 CULane 78.14 / 78.74 / 79.48 (small / medium / large; 220 / 152 / 58 FPS), TuSimple 97.01 / 96.98 / 97.24, CurveLanes 85.09 / 85.92 / 86.10 (RTX 2080).
- compute: FPS above.
- code: URL from memory, not opened: https://github.com/aliyun/conditional-lane-detection
- fit_to_us: reuse_idea - S - (a) lane END as a per-row existence classification (not a survival product); (b) continue / stop per RIM step is a discrete halting decision for forks (compare E94 events).
- compat_needs: row-wise parametrisation.
- compat_conflicts: one x per row.
- tried_before: E94 events continue / end / fork / merge.
- verified: checked

### CARD VS34 - UFLD: row-anchor classification with an explicit "no lane" cell
- papers: Ultra fast structure-aware deep lane detection (Qin, Wang, Li, ECCV 2020) https://arxiv.org/abs/2004.11757
- depth: M
- stage: S2, S3
- idea: Per lane and per predefined row the network classifies WHICH grid cell the lane crosses, with one extra class for "no lane in this row"; structural losses regularise neighbouring rows.
- formulation: output C x h x (w + 1); L_cls = sum CE over cell index; L_sim = sum_j |P_{i,j,:} - P_{i,j+1,:}|_1; L_shp = sum_j |(Loc_j - Loc_{j+1}) - (Loc_{j+1} - Loc_{j+2})|_1 with Loc = softmax expectation over the w cells (background cell excluded; differentiable, sub-cell); total L_cls + alpha L_str + beta L_seg, alpha = beta = 1, lambda = 1; auxiliary segmentation branch in training only; TuSimple rows 160..710 step 10, 100 cells; CULane rows 260..530 step 10, 150 cells; input 288 x 800. Output count 1.7e4 vs 1.15e6 for segmentation (CULane setting).
- input: global image_feat. output: per row cell index or none.
- training: CE; no matching.
- differentiable: yes (softmax expectation).
- sensors_data: images (TuSimple, CULane).
- results: TuSimple accuracy 96.06 % (R34, 5.9 ms), 95.87 % (R18, 3.2 ms); CULane F1 72.3 (R34, 175.4 FPS), 68.4 (R18, 322.5 FPS).
- compute: ms above.
- code: URL from memory, not opened: https://github.com/cfzd/Ultra-Fast-Lane-Detection
- fit_to_us: reuse_idea - S - per-row "no lane" = per-sample termination class; the second-order smoothness loss on expected positions is a cheap matching-free regulariser for sampled polylines.
- compat_needs: a monotone row axis.
- compat_conflicts: BEV lanes of arbitrary heading, forks, loops.
- tried_before: none.
- verified: checked

### CARD VS35 - GANet: keypoint confidence + offset to lane start, lane-aware aggregation along neighbours
- papers: A keypoint-based global association network for lane detection (Wang et al., CVPR 2022) https://arxiv.org/abs/2204.07335
- depth: M
- stage: S2, S6
- idea: Lane detection as keypoint estimation; each keypoint regresses an offset to its lane's start point (association in parallel); a lane-aware aggregator reads features at M = 9 neighbouring keypoints on the same lane with predicted offsets.
- formulation: confidence map (Gaussian splat, element-wise max on overlaps), penalty-reduced focal loss (alpha 2, beta 4, weight 1.0), quantisation L1 (1.0), start-point offset L1 (0.5), LFA auxiliary SmoothL1 with Hungarian matching of predicted to GT neighbour offsets (1.0); association: a keypoint joins a lane if its estimated start lies within theta_dis = 4 of that lane's start; LFA: F^(p_i) = sum_m w_m F(p_i + dp_i^m), M = 9; score threshold 0.4; 800 x 320; Adam 1e-3 poly, 300 epochs (TuSimple), 40 (CULane).
- input: image_feat / raster. output: keypoints + start offsets.
- training: dense + auxiliary hungarian_set (LFA only).
- differentiable: yes except grouping.
- sensors_data: images (TuSimple, CULane).
- results: CULane F1: S 78.79 (153 FPS), M 79.39, L 79.63 (63 FPS), curve F1 75.92 / 76.32 / 77.37; TuSimple F1 97.71 / 97.68 / 97.45; ablation (S): no LFA 77.84, LFA 78.30, LFA + aux 78.79.
- compute: FPS above.
- code: URL from memory, not opened: https://github.com/Wolfwjs/GANet
- fit_to_us: reuse_idea - S - "offset to a lane-level anchor" as a way to group sampled points into instances without Hungarian.
- compat_needs: start points per lane.
- compat_conflicts: lane graphs have several starts per path.
- tried_before: E42 / E45 learned link heads within 0.01 of the rules.
- verified: checked

### CARD VS36 - Anchor3DLane: 3-D line anchors projected to the image, iterative regression, per-point visibility
- papers: Anchor3DLane (Huang et al., CVPR 2023) https://arxiv.org/abs/2301.02371
- depth: M
- stage: S1, S2, S5
- idea: 3-D lane anchors (rays on the ground) are projected into the front-view feature map with the camera matrices, features are read at the projected points, class / x / z / visibility are regressed, and the regressed anchors are projected and read again.
- formulation: anchor = ray from (x_s, 0, 0) with pitch theta and yaw phi, N points at fixed y (N = 10 for ApolloSim and ONCE-3DLanes, 20 for OpenLane); x_s spaced 1.3 m (range not stated); yaw set {0, +-1, +-3, +-5, +-7, +-10, +-15, +-20, +-30} degrees (17 values, DENSER NEAR ZERO), pitch set {0, +-1, +-2, +-5} degrees (7 values): 119 anchors per x_s; projection [u~, v~, d]^T = K T_{g->c} [x, y, z, 1]^T, u = (W_f / W) u~ / d, v = (H_f / H) v~ / d; front-view feature 45 x 60 x 64 from a 360 x 480 input; bilinear read, features of an anchor's points concatenated; iterative regression re-projects predictions as new anchors; positives = the 3 nearest anchors per GT lane (visibility-weighted distance; thresholds not stated); focal (alpha 0.5, gamma 2) + L1 on x / z + L1 on visibility, lambda_cls = lambda_reg = 1; NMS distance 2; equal-width constraint as post-processing (F1 54.8 -> 55.0).
- input: image_feat + camera + point_set (anchors). output: 3-D polylines with visibility.
- training: nearest-anchor assignment (matching_free).
- differentiable: yes except NMS.
- sensors_data: monocular camera (OpenLane, ApolloSim, ONCE-3DLanes).
- results: OpenLane val F1 53.1 (x error close / far 0.300 / 0.311 m), 53.7 with iterative regression, 54.3 with temporal; PersFormer 50.5; ablation OpenLane-300: iterations 1 / 2 / 3 F1 54.8 / 56.3 / 57.0 (the row mapping is ambiguous in the text); front-view sampling 53.1 vs BEV-feature sampling 47.6.
- compute: not stated.
- code: URL from memory, not opened: https://github.com/tusen-ai/Anchor3DLane
- fit_to_us: reuse_idea - S - (a) a NON-UNIFORM yaw fan, dense near forward, is a published precedent for a teardrop-like angular distribution; (b) per-point visibility trained with L1 (VS43); (c) two to three re-sampling iterations gain +1.5 / +2.2 F1.
- compat_needs: camera projection for the reads.
- compat_conflicts: perspective features; on a cached BEV raster the projection collapses to a direct grid_sample.
- tried_before: none.
- verified: checked

### CARD VS37 - CurveFormer: curve queries as ordered anchor points, deformable attention at projected points
- papers: CurveFormer (Bai et al., arXiv 2022) https://arxiv.org/abs/2209.07989
- depth: M
- stage: S2, S5
- idea: A query is a 3-D curve (confidence + two polynomials + start / end) whose positional part is an ordered set of anchor points at fixed Y positions; anchors are projected into multi-scale image features, bilinearly read and aggregated by deformable attention; anchors are refined layer by layer.
- formulation: 10 fixed Y positions between 5 and 100 m (inferred from the implementation statement); a binary indicator marks projected points that fall outside a feature level (Eq. 3) - an explicit "outside the map" flag; offsets from an MLP on the concatenated anchor feature and query; K and bounds not stated; per layer (dx, dz) offsets by a shared linear layer; 4 decoder layers best (F 94.1 vs 93.1 / 93.2 / 94.0 / 93.6 for 2 / 6 / 8 / 10 on the rarely-observed split); Hungarian, weights 2 / 5 / 2 / 2; EfficientNet, 360 x 480, range [-30, 30] x [3, 103] x [-10, 10] m; Adam 2e-4, 100 epochs.
- input: image_feat + camera + queries. output: 3-D polylines.
- training: hungarian_set.
- differentiable: partial.
- sensors_data: monocular camera (OpenLane).
- results: OpenLane F-score 50.5 (up-down 45.2, curve 56.6, night 49.1, merge and split 45.4); x error near / far 0.340 / 0.772 m, z 0.207 / 0.651 m; no ablation on the number of curve points, no FPS.
- compute: not stated.
- code: not checked.
- fit_to_us: reuse_idea - S - (a) curve-ordered anchor points as the query positional part; (b) the explicit per-level in-range indicator for projected points.
- compat_needs: camera projection, Hungarian.
- compat_conflicts: perspective features.
- tried_before: none.
- verified: checked

### CARD VS38 - LATR: lane-aware queries (lane x point) with a dynamic ground-plane positional embedding
- papers: LATR (Luo et al., ICCV 2023) https://arxiv.org/abs/2308.04583
- depth: M
- stage: S2, S5
- idea: Queries are the broadcast sum of a lane-level embedding (pooled from image features by instance activation maps) and a point-level embedding per predefined height; a ground-plane grid refined layer by layer gives a 3-D positional embedding for the image features, so no BEV transform exists.
- formulation: Q = Q_lane (+) Q_point (N x M queries, M not stated); plane initialised horizontally at z = -1.5, projected into the image, scattered to a canvas, MLP -> positional embedding; each decoder layer predicts (d theta_x, d h) of the plane; 6 layers (LATR-Lite 2), deformable attention 4 heads x 8 points, 256-d; losses w_s = 5 (segmentation), w_p = w_l = 1; lane loss: x 2.0 L1, z 10.0 L1, class 10.0 focal (gamma 2, alpha 0.25), visibility BCE 1.0; dice-based bipartite matching reused for the predictions.
- input: image_feat + camera. output: 3-D polylines with visibility.
- training: hungarian_set.
- differentiable: partial.
- sensors_data: monocular camera (OpenLane).
- results: OpenLane val F1 61.9 vs PersFormer 50.5; x error near / far 0.219 / 0.259 m; Lite F1 61.5; ablation OpenLane-300: lane-only queries 66.5 vs lane + point queries 70.4 F1; ground embedding none 67.9, fixed frustum 69.1, fixed plane 69.2, dynamic plane 70.4.
- compute: FPS not stated.
- code: URL from memory, not opened: https://github.com/JMoonr/LATR
- fit_to_us: reuse_idea - S - lane x point factorised queries (each sample point has its own embedding) and a per-point visibility BCE.
- compat_needs: camera model, Hungarian.
- compat_conflicts: perspective features.
- tried_before: none.
- verified: checked

### CARD VS39 - PolarMask: rays from a centre with regressed lengths and the Polar IoU loss
- papers: PolarMask (Xie et al., CVPR 2020) https://arxiv.org/abs/1909.13226
- depth: M
- stage: S2, L
- idea: An instance mask is a centre plus n rays at fixed angles; only the ray lengths are regressed; a loss built on the ratio of the sum of minima to the sum of maxima of predicted and target lengths approximates mask IoU.
- formulation: n = 36 rays at 10 degree spacing; contour point x_i = cos(theta_i) d_i + x_c, y_i = sin(theta_i) d_i + y_c; Polar IoU = sum d_min / sum d_max, loss = log(sum d_max / sum d_min) (the squared form changes results by +-0.1 mAP); polar centerness = sqrt(min d / max d), multiplied with the class score at inference; centre samples within 1.5 x stride of the mass centre (about 9 - 16 pixels per instance); a ray that hits the contour several times takes the maximum length; a ray from an outside centre that never meets the contour gets eps = 1e-6; FCOS-style head, box branch not needed.
- input: image_feat levels. output: point_set (n ray lengths per location).
- training: dense centre sampling, no Hungarian.
- differentiable: yes.
- sensors_data: images (COCO).
- results: COCO minival ResNet-50: rays 18 / 24 / 36 / 72 -> AP 26.2 / 27.3 / 27.7 / 27.6; Polar IoU loss 27.7 vs best smooth-L1 25.1 (Table 1b); adding polar centerness raises 27.7 to 29.1 (Table 1c); test-dev 32.9 AP (ResNeXt-101). Limits named: concave and donut-shaped instances (neither centre lies inside), multiple intersections.
- compute: not stated.
- code: URL from memory, not opened: https://github.com/xieenze/PolarMask
- fit_to_us: reuse_idea - S - evidence for the number of rays (18 -> 36 gains 1.5 AP, 72 adds nothing) and for a scale-free ratio loss on lengths; our teardrop uses 12 rays, below the smallest tested count, but serves forks, not mask area.
- compat_needs: a centre.
- compat_conflicts: closed star-shaped targets; a lane graph is open and branching.
- tried_before: none.
- verified: checked

### CARD VS40 - PolarFormer: polar BEV from image columns, multi-scale polar maps, polar reference points
- papers: PolarFormer (Jiang et al., AAAI 2023) https://arxiv.org/abs/2206.15398
- depth: M
- stage: B, S2
- idea: Each camera sees a wedge, so the BEV is a (radius x azimuth) grid; every image column is attended into one polar ray by cross-attention; the decoder uses deformable attention with reference points in cylindrical coordinates.
- formulation: multi-scale polar maps (radius, azimuth) = (64, 256), (32, 128), (16, 64); best single setting N_theta = 256, R = 64 (39.6 mAP / 45.8 NDS); azimuth phi = arctan((x_I - u0) / f_x), radius rho = z sqrt(((x_I - u0)/f_x)^2 + 1); 3 cross-plane layers per scale; alignment samples polar points into each camera with a binary visibility weight; 6 encoder + 6 decoder layers; offsets (d rho, d phi, d z). The authors' argument: Cartesian BEV downsamples near regions and upsamples far regions without adding information.
- input: image_feat from six cameras. output: 3-D boxes.
- training: hungarian_set (decoder).
- differentiable: partial.
- sensors_data: nuScenes cameras.
- results: val R101 39.6 mAP / 45.8 NDS; ablation: Cartesian / Cartesian 38.1 / 45.5, polar features + Cartesian prediction 38.5 / 45.0, polar / polar 39.6 / 45.8; single-scale polar 39.1 / 45.0; a loss computed in polar coordinates failed (31.5 / 38.2, attributed to the 0 / 2 pi discontinuity) so the loss stays Cartesian; fixed sinusoidal positional encoding best (39.6) vs learned 2-D 38.8.
- compute: parameters not stated.
- code: URL from memory, not opened: https://github.com/fudan-zvg/PolarFormer
- fit_to_us: not_suitable (image-to-polar encoder needs perspective features); reuse_idea - S - resolution decaying with range, and the lesson to compute losses in Cartesian metres.
- compat_needs: images.
- compat_conflicts: cached Cartesian raster.
- tried_before: none.
- verified: checked

### CARD VS41 - PolarBEVDet: polar BEV grid with equal cell budget, polar-aware temporal alignment
- papers: PolarBEVDet (Yang et al., arXiv 2024) https://arxiv.org/abs/2408.16200 ; related polar adaptive BEV (Yang et al., ICRA 2023) https://arxiv.org/abs/2304.00525 (search summary only)
- depth: M (PolarBEVDet), A (adaptive BEV)
- stage: B, S8, S2
- idea: Lift-splat into a polar grid (azimuth x radius) with the same cell count as a Cartesian one, so cells are dense near the ego and sparse far away; temporal fusion converts polar cells to Cartesian, applies ego motion, converts back and samples bilinearly.
- formulation: azimuth [-pi, pi] in N_theta bins, radius [r_min, r_max] in N_r bins (values not stated); default 256 x 64 = 16,384 cells (large backbones 384 x 96), same as a 128 x 128 Cartesian grid; centre-based head predicting (theta, r, z, w, l, h, alpha_p = alpha_c - theta, v_theta, v_r); alignment: theta_t = -pi + i delta_theta, r_t = r_min + j delta_r, x = r cos theta, y = r sin theta, ego-motion transform, back to polar indices, bilinear sampling, 1x1 conv fusion.
- input: image_feat from six cameras. output: 3-D boxes.
- training: heat-map targets (no Hungarian).
- differentiable: yes (bilinear sampling).
- sensors_data: nuScenes cameras.
- results: val R50 256 x 704: baseline 0.496 NDS / 0.397 mAP; + polar 0.509 / 0.414; + 2-D auxiliary supervision 0.521 / 0.429; + SAE 0.527 / 0.434; under an azimuth revolve test Cartesian drops 2.4 NDS, polar 0.4; porting to BEVDet 0.360 -> 0.382 NDS, BEVDet4D 0.426 -> 0.450, BEVDepth 0.457 -> 0.474. Text and Table I disagree on the gain over the best baseline (1.9 % mAP / 2.1 % NDS in the text vs +2.3 mAP / +1.9 NDS in the table); no measured speed or memory comparison.
- compute: none stated.
- code: URL from memory, not opened: https://github.com/Yzichen/PolarBEVDet
- fit_to_us: reuse_idea - S - the cell-budget argument (dense near, sparse far) and the polar -> Cartesian -> ego motion -> bilinear recipe, the same as our ego-warping by grid_sample.
- compat_needs: raster.
- compat_conflicts: none.
- tried_before: E41 temporal feature fusion (+0.008 F1).
- verified: checked

### CARD VS42 - Pyramid level assignment by region size (FPN)
- papers: Feature pyramid networks for object detection (Lin et al., CVPR 2017) https://arxiv.org/abs/1612.03144
- depth: M
- stage: S2
- idea: Which pyramid level a region reads from is a function of its size: RoIs of area w * h use level k = floor(k0 + log2(sqrt(w h) / 224)) with k0 = 4 (P2..P5; P6 only for RPN anchors).
- formulation: top-down pathway with nearest-neighbour 2x upsampling, 1x1 lateral conv, element-wise add, 3x3 smoothing conv, 256 channels; RPN anchors one scale per level 32^2..512^2, three ratios.
- input: backbone pyramid. output: feature pyramid.
- training: standard detection; none matching.
- differentiable: yes.
- sensors_data: images (COCO).
- results: COCO minival ResNet-50: RPN AR1k: C4 baseline 48.3, finest level only 51.3, full pyramid 56.3; Fast R-CNN AP 31.9 / 33.4 / 33.9; top-down without lateral connections 46.1 AR1k; bottom-up pyramid only 49.5 (24.9 AP).
- compute: not stated.
- code: not checked.
- fit_to_us: reuse_idea - S - a size-to-level rule: replace object size by ring spacing or tip uncertainty to decide how much of f0 / f1 / f2 to read at ring k (own mapping; no paper does this by DISTANCE FROM THE EGO for centre-lines).
- compat_needs: pyramid.
- compat_conflicts: none.
- tried_before: E94 uses the pyramid f0 / f1 / f2 (fixed assignment).
- verified: checked

### CARD VS43 - Visibility / observability masks for censoring (Occ3D, ray-cast occlusion, per-point visibility heads)
- papers: Occ3D (Tian et al., NeurIPS 2023 datasets track, from memory) https://arxiv.org/abs/2304.14365 ; Bird's-eye-view panoptic segmentation using monocular frontal view images (PanopticBEV, Gosala, Valada, RA-L 2022) https://arxiv.org/abs/2108.03227 ; Parametric depth based BEV feature learning (Yang et al., arXiv 2023, abstract only) https://arxiv.org/abs/2307.04106 ; GaussianCaR (arXiv 2602.08784, ICRA 2026 per the search summary; the paper itself was not opened) https://arxiv.org/abs/2602.08784 ; per-point visibility heads in Anchor3DLane (VS36) and LATR (VS38)
- depth: M (Occ3D, PanopticBEV), A (the others)
- stage: S3, D, E
- idea: How to say "the model could not see this" so it is not counted as "the lane ended". (1) Occ3D: LiDAR visibility = a voxel is visible if it reflects a point or a ray passes through it; free = traversed only; unobserved = the rest; camera visibility = a ray from each occupied voxel to the camera origin, voxels behind the first occupied voxel are unobserved, voxels reached by no camera ray unobserved; evaluation counts only observed voxels. (2) PanopticBEV: a BEV pixel is occluded if it is lower than a previously seen pixel along a 2-D ray cast from the camera across the BEV image (built from the predicted height map); occlusion becomes an explicit stuff CLASS to predict ("extremely difficult to hallucinate labels behind occlusions"), pixels outside the field of view are zeroed and cropped; the page does not say whether occlusion pixels are ignored in the loss or evaluation. (3) The parametric-depth paper proposes a visibility-aware evaluation metric (abstract; how the visibility is obtained is not stated there). (4) Trained per-point visibility: Anchor3DLane (L1, weight 1) and LATR (BCE, weight 1.0); OpenLane labels carry a visibility attribute (from memory). (5) The foundation: GaussianCaR is reported on nuScenes with IoU 57.3 % vehicles, 82.9 % roads, 50.1 % lane dividers (search summary), so a vehicle BEV output exists in the published model; whether the cached 7 map-layer logits contain a vehicle layer is not known to this cluster.
- formulation: Occ3D grid 200 x 200 x 16 at 0.4 m, +-40 m, z from -1 to 5.4 m; virtual camera points at depth 1e3; PanopticBEV BEV image 768 x 704 (KITTI-360) / 896 x 768 (nuScenes) at 7.4 / 7.6 cm per pixel; no further thresholds on the pages.
- input: occupancy labels (from lidar) or a height map + camera origin. output: binary masks or an occlusion class.
- training: labels / evaluation masks; PanopticBEV learns the occlusion class.
- differentiable: not applicable.
- sensors_data: nuScenes lidar + cameras (Occ3D); camera only (PanopticBEV: KITTI-360, nuScenes); camera + radar (GaussianCaR).
- results: none copied.
- compute: not stated.
- code: URL from memory, not opened: https://github.com/Tsinghua-MARS-Lab/Occ3D
- fit_to_us: reuse_idea - S - (a) a static camera-coverage mask (field of view, minimum distance) needs only the calibrations; (b) an occluder mask from a vehicle / object layer (if the foundation or a small head provides one) combined with a ray rule from the cameras or the ego position, in the PanopticBEV style, gives a "shadow" mask without lidar; (c) lidar-derived visibility could build censoring LABELS offline, but that brings lidar into the label pipeline (a project decision); (d) a learned observability head with BCE as in LATR, or an explicit "occluded" class as in PanopticBEV.
- compat_needs: a mask or class per sample position.
- compat_conflicts: camera + radar only at inference - the mask must be predicted or purely geometric.
- tried_before: none.
- verified: partly

### CARD VS44 - Lane extent / termination heads across lane detectors (synthesis card)
- papers: see VS26 (CLRNet), VS27 (LaneATT), VS33 (CondLaneNet), VS34 (UFLD), VS35 (GANet), VS36 (Anchor3DLane), VS38 (LATR)
- depth: M (through the cards)
- stage: S3, S7
- idea: Every method seen decides where a lane stops with a regressed length or a per-row / per-point binary signal; none uses a cumulative survival product and none separates "lane ends" from "not observable".
- formulation: LaneATT and CLRNet: length l (end index = start index + l - 1; smooth-L1); CondLaneNet: per-row BCE "lane passes this row" (weight 1.0) plus RIM continue / stop; UFLD: extra (w + 1)-th class per row; Anchor3DLane: visibility per point (L1); LATR: visibility per point (BCE); GANet: lane ends where keypoints stop (no explicit end).
- input / output / training / differentiable: see the cards.
- sensors_data: images; OpenLane 3-D.
- results: none (synthesis).
- compute: not stated.
- code: see the cards.
- fit_to_us: reuse_idea - S - per-row BCE existence is a non-cumulative hazard-like signal; length regression is a single-number endpoint.
- compat_needs: a monotone axis.
- compat_conflicts: forks.
- tried_before: E94 end event.
- verified: checked

## 3. Cluster compatibility notes

### 3a. Interface contracts
| slot | what flows in the literature | mapping to our cached BEV pyramid and vertex / token / validity / edge output |
|---|---|---|
| S2 reader | reference points (+ learned offsets, softmax weights) read from image_feat or bev_raster: Deformable DETR, DAB-DETR, LaneSegNet, TopoBDA, BezierFormer, LATR, CurveFormer; fixed point sets along a hypothesis: L-CNN / HAWP 32, CLRNet / Polar R-CNN 36, DeepLSD 50 (10 for refinement), LaneATT 72 rows, Anchor3DLane 10 / 20, BezierFormer 5 x 5; window lookup: RAFT 81 per level x 4 levels | bilinear `grid_sample` of the f0 / f1 / f2 pyramid at ego-metre positions (zero padding outside); no camera model needed. The teardrop is 12 rays x 5 rings = 60 positions per tip (project numbers), i.e. between E94's 34 reads and RAFT's 324 per pixel; with three pyramid levels each position yields 64 + 96 + 128 channels |
| S3 verifier | one score per hypothesis (L-CNN, HAWP, CLRNet / LaneATT class head); per-row or per-point existence (CondLaneNet, UFLD, Anchor3DLane / LATR visibility); PMF or hazard per interval (survival models); fraction of agreeing samples (DeepLSD); support count (ScaleLSD); centerness (ELSD) | per-ring conditional probability nu_k, survival T_k = prod_{j<=k} nu_j, endpoint law P(k) = T_k (1 - nu_{k+1}), P(K) = T_K = censored at range; validity = T x direction prior |
| S4 selector | pair-then-verify and threshold (L-CNN, HAWP), top-K of local proposals (Polar R-CNN), SimOTA / dynamic top-k (CLRNet, BezierFormer) | b >= 2 endpoints with gradient through the selected items |
| S5 refiner | residual on a hypothesis with a per-stage loss: Cascade R-CNN, Deformable DETR, DINO, CLRNet, RAFT, Anchor3DLane | stage B / C with per-stage radius |
| token | pooled sample features before the FC (L-CNN: 128 x 8), query embedding (DETR family), RoI feature (Polar R-CNN) | our 128-d step token can be the ring-feature summary |

Survival algebra written once (own notation; ties the cards together): with nu_j the conditional probability that sample j is still on the lane given j-1 was, T_k = prod_{j<=k} nu_j, hazard h_j = 1 - nu_j. P(last verified = k) = T_k (1 - nu_{k+1}) = h_{k+1} prod_{j<=k}(1 - h_j) = DRSA's p_l = PonderNet's p_n = NeRF's w_i (alpha_i = h_i, T_i = prod_{j<i}(1 - alpha_j)). With a ray stopped by a failure at sample f: log L = sum_{j<f} log nu_j + log(1 - nu_f); right-censored after sample c: log L = sum_{j<=c} log nu_j (VS01 conventions). Per-metre version: nu_j = exp(-sigma_j delta_j) (NeRF, piecewise-constant hazard, constant-hazard interpolation), which makes unequal gaps 4 / 4 / 4 / 6 / 6 m consistent. Expected-loss version (PonderNet): sum_k P(k) l(endpoint_k, GT) + beta KL(P || geometric prior).

### 3b. Known adapters and conflicts
1. Conditional versus marginal labels: the hazard BCE for sample k is applied only while all earlier samples were alive (pycox cumsum up to the event / censoring index). A classifier trained on "sample k is near a lane" for all k is the wrong object for the product.
2. Censoring convention must be written down (VS01): Nnet-survival rounds the censoring time to the nearest interval border; logistic-hazard counts the censoring interval as survived; PMF form excludes it; PC-hazard weights the last interval by the observed fraction; PonderNet gives the leftover mass to the last step; NeRF never defines the last delta; the raycaster of Agro et al. adds a virtual stop at the measured depth beyond the grid in training (appendix: mass placed at the boundary) and clamps rays at evaluation. For a ray that leaves the BEV range the natural choice is "alive through the last in-range ring" (no event term).
3. Occlusion is not the same as range (own derivation, not from a paper): an occluded stretch is unobserved, the lane there usually continues, and the rings behind it are conditional on surviving it. Three published ways to represent "unobserved": a mask that removes the sample from the evaluation (Occ3D evaluates only observed voxels), an explicit occlusion class to predict (PanopticBEV), a per-point visibility output trained with L1 / BCE (Anchor3DLane, LATR). For the ray: mask the hazard loss on unobserved samples, keep the survival chain, and give the model an observability flag as input so that it can fall back on the prior hazard. CurveFormer already carries an explicit per-level in-range indicator in its sampling step.
4. Zero padding of `grid_sample` makes out-of-range reads look like "no lane" (CLRNet and LaneATT pad with zeros, Deformable DETR reference code uses zeros) - without a censoring flag the verifier learns "edge of map = lane ends".
5. Hungarian dependence: Deformable DETR, DINO (main queries), DAB-DETR, Sparse R-CNN, LETR, BezierLaneNet, LaneSegNet, TopoBDA, CurveFormer, LATR, GANet (auxiliary) need bipartite matching - conflict with the matching-free rule. Matching-free or threshold alternatives found: Cascade R-CNN IoU thresholds, CLRNet dynamic top-k (k = 1 NMS-free), LaneATT / Anchor3DLane distance thresholds, Polar R-CNN SimOTA (O2M), BezierFormer SimOTA top-1, CondLaneNet / UFLD / RAFT / PolarMask dense targets.
6. Row-wise parametrisations (UFLD, CondLaneNet, LaneATT, CLRNet offsets, BezierLaneNet column pooling) assume one x per row and a mostly vertical lane: they conflict with BEV graphs of arbitrary heading, forks and loops.
7. Perspective-specific parts (Anchor3DLane projection, CurveFormer, LATR, PolarFormer cross-plane encoder, PolarBEVDet lift) do not apply to a cached BEV raster; their anchor / query / loss ideas transfer. Anchor3DLane's own ablation: front-view sampling 53.1 F1 vs BEV-feature sampling 47.6 - for a learned view transform, not for a BEV that is given.
8. Bezier <-> polyline: Bernstein matrix at fixed t (BezierLaneNet), control points as reference points (TopoBDA), five reference points at t in {0, 0.25, 0.5, 0.75, 1} (BezierFormer). A cubic cannot represent a branching ray.
9. Query <-> bilinear sample: reference point + K offsets + softmax weights (Deformable DETR family) equals "pattern + learned offsets + attention weights over the pattern"; offsets are in pixels of the level (divide by the level cell size to get metres: 0.5 / 1 / 2 m).
10. Polar loss pitfall: PolarFormer's loss in polar coordinates failed (31.5 / 38.2 vs 39.6 / 45.8) because of the 0 / 2 pi discontinuity; compute all losses in Cartesian metres, as the project already does.
11. Stop-gradient: RAFT, Deformable DETR, Sparse R-CNN, CLRNet and Polar R-CNN all cut a gradient somewhere (table below); "no detach between stages" is supported only in the DINO form (undetached previous position inside the next stage's prediction, +0.4 AP).

### 3c. Combinations the papers themselves demonstrate
- L-CNN: junction proposals + pair sampler + LoI verifier; HAWP replaces the proposals by an attraction field and needs only the natural negative sampler; ELSD, LETR and ScaleLSD later drop the sampled verifier entirely (Wireframe sAP10: L-CNN 62.9, HAWP 66.5, LETR 65.2, ELSD 68.9 at 42.6 FPS), which suggests that sampled verification matters when proposals are many and uninformative.
- CLRNet: FPN levels in sequence + ROIGather + Line-IoU + dynamic top-k (refinement 52.80 -> 54.74 mF1, ROIGather +0.5).
- DINO: contrastive denoising + mixed query selection + look forward twice (47.0 -> 47.4 -> 47.9 AP in the ablation).
- Deformable DETR: iterative refinement + two-stage proposals (+1.6, +0.8 AP).
- Polar R-CNN: local polar proposals + global sampled head + O2M / O2O with stopped gradient (71.68 -> 80.81).
- LaneSegNet: lane attention + identical initialisation + heads-to-regions (28.8 -> 32.6 mAP); TopoBDA: BDA + instance masks + one-to-many decoder (46.5 -> 48.0 OLS_l) + radar / SD-map fusion in BEV.
- Sparse4D v3: instance denoising + decoupled attention + quality head; RenderOcc: 2-D rendering + auxiliary rays + weighted ray sampling (+3D labels 26.11 vs 23.93 mIoU).
- Survival heads combined with rendering-style weights: DRSA combines event-time likelihood and status cross-entropy; PonderNet combines expected loss and a geometric prior.

### 3d. Evidence tables
Detach / stage supervision (as seen on the pages):
| method | what is cut | stage losses | evidence |
|---|---|---|---|
| Cascade R-CNN | not stated | per stage | cascade 38.9 vs iterative 35.4 AP |
| Deformable DETR | gradient blocked at the previous box | auxiliary per layer | refinement 45.4 vs 43.8 AP (cut itself not ablated) |
| DINO | box passed on is detached, but the prediction uses the undetached previous box | per layer | look forward twice +0.4 AP (47.0 -> 47.4) |
| Sparse R-CNN | gradient blocked at proposal boxes each stage | not stated | 6 stages 42.3 vs 3 stages 39.9 AP |
| RAFT | `coords1` detached every iteration, gradient only through Delta f | all iterations, gamma^{N-i} | more updates at test time keep helping: 8 -> 32 updates, Sintel clean EPE 1.61 -> 1.43, no divergence up to 200 |
| CLRNet (code) | priors detached before pooling | every stage | 3 stages 54.74 vs 1 stage 52.80 mF1 |
| Polar R-CNN | O2O branch gradient stopped | - | 71.68 -> 80.81 F1@50 |
| Anchor3DLane | not stated | - | 1 / 2 / 3 iterations 54.8 / 56.3 / 57.0 F1 |
Reading: everyone detaches the geometry that decides the next read; only DINO feeds credit back, through the loss, not through the sampling; nobody reports full no-detach on point positions.

Learned offsets (bounds, init, learning rate):
| method | bound | init | other |
|---|---|---|---|
| Deformable DETR | none (code: no clamp), offsets in level pixels | weight 0, bias = 8 directions x (k + 1) | lr x 0.1 for offsets and reference points |
| DAT | s tanh(.), s = 2 | not stated | sweep s 0..16 stable |
| Sparse4D | sigmoid - 0.5 in box units (+-0.5) | not stated | scaled by box size, rotated by yaw |
| LaneSegNet | not stated | K = 32 evenly around each reference point (4 distances x 8 directions) | identical init across heads in layer 1 |
| TopoBDA | not stated | K = 32 per control point | - |
| BezierFormer | not stated | 5 reference x 5 samples | - |
| CurveFormer, LATR | not stated | - | 4 heads x 8 points (LATR) |

Number of rays / points (what was tested): PolarMask rays 18 / 24 / 36 / 72: 26.2 / 27.3 / 27.7 / 27.6 AP; LaneATT anchors 250 ... 1250: 68.68 / 75.45 / 75.80 / 76.66 / 75.91 F1; BezierFormer reference points 1 -> 5: +1.5 to +1.7 F1 (CurveLanes 87.60 -> 89.06, OpenLane 52.13 -> 53.84); LoI sample count: not ablated in L-CNN or HAWP; Sparse R-CNN stages 1 / 2 / 3 / 6 / 12: 21.7 / 36.2 / 39.9 / 42.3 / 41.6 AP.

### 3e. The teardrop against polar and anchor designs
- Equiangular rays with a regressed length (PolarMask, 36 rays, saturating between 36 and 72) are the nearest published ray design; ours have 12 rays, anisotropic length (short in all directions, long forward) and a length obtained from verification, not regression.
- Anchor3DLane uses a non-uniform yaw fan (finer near zero: 0, +-1, +-3, +-5, +-7, +-10, +-15, +-20, +-30 degrees); LaneATT / CLRNet / Polar R-CNN use straight line priors (start + angle) and read 36 - 72 equally spaced points along each prior's own length. None uses common radii for all rays, which is what the teardrop's shared rings 4 / 8 / 12 / 18 / 24 m do; with common radii the survival of different rays can be compared ring by ring and the ring features can share one scale choice per radius.
- DAB-DETR (width / height modulation), Sparse4D (pattern scaled by instance size, bounded offsets) and Deformable DETR (fixed fan + zero-initialised offsets) are the three published recipes for "reference pattern + gain + learned deviation".
- Polar grids (PolarFormer, PolarBEVDet) change the resolution with range at equal cell count; our cached pyramid does the same through levels, so far rings can be read from f2 (2 m cells) and near rings from f0 (0.5 m cells) - CLRNet's deepest-first order beat the reverse by 1.2 mF1.

### 3f. Verifying long lines: intermediate samples versus one score
- L-CNN / HAWP pool 32 samples into one score (max-pool 32 -> 8, FC); the number of samples is not ablated in either paper; no localisation of the failure along the line.
- DeepLSD counts agreeing samples (50 samples, more than 50 % inliers) and refines on 10; ScaleLSD counts supporting pixels; ELSD and LETR score without sampling. So the line-detection literature offers: pooled-and-learned (L-CNN, HAWP), counted (DeepLSD, ScaleLSD), none (ELSD, LETR). A cumulative per-sample product with an endpoint law is new relative to all of them (the earlier review's claim stands).
- In lane detection the strongest effect among these is line-aligned pooling itself (LaneATT: without it 64.89 vs 76.68 F1).

### 3g. What transfers to a cached BEV raster pyramid (bilinear reads, no camera model)
- Transfers as published (the head only needs `grid_sample` on a feature map): L-CNN / HAWP LoI pooling (VS12, VS13), DeepLSD-style sampled checks (VS14), CLRNet ROI pooling, ROIGather and Line-IoU (VS26), LaneATT anchor pooling (VS27, with line anchors), Polar R-CNN second-stage sampling (VS28), LaneSegNet lane attention and TopoBDA / BezierFormer curve reads (VS29, VS31, VS32; they already run on a 0.5 m BEV raster or its image-feature equivalent), Deformable DETR / DAB-DETR / DAT readers (VS20, VS23, VS25), RAFT window lookups (VS21), Sparse4D keypoint patterns without the projection (VS24), the cascade and look-forward-twice recipes (VS18, VS19), all survival / hazard losses (VS01 - VS04, VS07).
- Needs an adapter: Anchor3DLane, CurveFormer, LATR (drop the camera projection; anchors and queries already live in ground coordinates, so reads become direct `grid_sample` at ego metres; their own ablation says front-view reads beat BEV reads of a learned view transform, which does not apply to a given BEV); row-wise decoders (UFLD, CondLaneNet, BezierLaneNet column pooling) only after replacing "row" by "arc length along the hypothesis".
- Not transferable: PolarFormer / PolarBEVDet lifts (perspective image features to a polar grid), UniPAD / RenderOcc / Vampire rendering from camera rays or a trainable height-resolved volume, Khurana-style lidar raycasting; only their loss and weighting ideas carry over.

### 3h. Candidate mixes for systematic testing (each element has paper evidence; the combinations themselves are untested and are the author's suggestion)
| mix | elements | what it tests | known obstacle |
|---|---|---|---|
| M1 survival ray | VS01 masked BCE + censoring flag, VS04 expected loss over P(k) + geometric KL, VS07 distortion on P(k) | endpoint law trained without argmax; bimodality control | conditional labels; censoring convention (3b.1 - 3b.3) |
| M2 reader | VS20 zero-init fan + lr x 0.1, VS24 sigmoid - 0.5 bounded offsets scaled by (length, width), VS23 gain modulation; alternative VS21 window lookup | learned deviation from the fixed teardrop without leaving the map | offsets in level cells (0.5 / 1 / 2 m); zero padding at the border |
| M3 cascade | VS18 radius schedule per stage, VS19 look-forward-twice residual, VS26 loss on every stage with detached geometry, VS21 gamma-weighted sequence loss | whether stage-wise tightening and credit feedback help the refinement beyond E39 / E120 | E39 / E120 already small gains |
| M4 verifier baselines | VS12 single-score LoI head (32 samples; or 5 / 12 / 24), VS14 inlier fraction of heat map and direction | whether the cumulative per-ring product beats one score and a free test | sample-count ablation absent in the literature |
| M5 curve reads | VS29 five reference points along the curve (t = 0 .. 1), VS31 identical initialisation, VS32 K = 32 per control point | stage B reads along chord / Hermite / Bezier instead of a free teardrop | cubic Bezier cannot branch; Hungarian-based decoders |
| M6 censoring | VS43 (in-range flag, camera-coverage mask, optional occluder mask, optional learned observability) + VS01 | survival not punished for range or occlusion | no occluder mask known for camera + radar |
| M7 scale | VS42 size-to-level rule mapped to ring spacing, VS26 deepest-first order | which pyramid level each ring reads | no distance-based precedent |

## 4. Reuse versus re-implement
- Re-implement (S, from the method sections; no external code needed): hazard NLL with masks (pycox pattern: cumulative BCE up to the index), conditional-label construction for rays, censoring flags, expected-loss + geometric-prior KL (VS04), distortion regulariser (VS07), single-score LoI baseline (32 samples, 1x1 conv, max-pool, FC) (VS12), DeepLSD-style inlier-fraction baseline (VS14), cascade radius schedule (VS18), look-forward-twice residual (VS19), bounded offsets s tanh / sigmoid - 0.5 with zero init and lr x 0.1 (VS20, VS24, VS25), RAFT-style multi-level window lookup (VS21), Bernstein sampler (VS30), polar-IoU-like ratio loss (VS39).
- Adapt code (M): the pure-PyTorch deformable-attention core of Deformable DETR (copy `ms_deform_attn_core_pytorch`, do not import the CUDA module); CLRNet's ROIGather / `pool_prior_features` pattern (only `grid_sample` calls; the repo's framework dependencies were not checked); LaneSegNet's lane attention only as an idea (mmcv / mmdet3d codebase).
- Baseline only: L-CNN / HAWP-style verifier; CLRNet-like head.
- Not suitable: LETR, Sparse R-CNN and the main queries of DINO / DAB-DETR (Hungarian), PolarFormer / PolarBEVDet encoders (perspective features), UniPAD / RenderOcc / Vampire / Khurana (trainable 3-D volumes or lidar), UFLD / CondLaneNet row-wise decoding.

## 5. Open questions and gaps
- No perception paper trains a discrete-time hazard along a ray with censoring from range or occlusion; the medical papers give the likelihood but no spatial analogue, and none studies calibration of a product of 5 conditional probabilities (if each nu is 0.95 the product at ring 5 is 0.77; calibration of P(k) is untested anywhere read).
- The LoI sample count (32) is never ablated; whether 5 rings are enough for a 24 m ray, versus 12 or 24, has to be measured.
- Bounding offsets: only DAT sweeps the bound scale (stable from 0 to 16, no with / without comparison); no lane or centre-line paper ablates bounds.
- Detach versus no-detach for point positions: only DINO (boxes, +0.4 AP); no lane or polyline evidence.
- Stage-wise tightening of the positive radius for polylines: no paper; CLRNet varies the feature level, Cascade R-CNN varies the IoU threshold for boxes.
- Level selection by distance from the ego for centre-lines: no paper (FPN rule is by region size).
- Visibility for camera + radar only: no method; Occ3D masks need lidar labels; PanopticBEV builds its occlusion mask from a predicted height map and a camera ray rule and does not say whether the loss ignores it; how the parametric-depth paper obtains visibility is not stated on the page seen. Whether the cached foundation features give a usable vehicle / occluder layer is not known here (GaussianCaR is reported on vehicles, roads and lane dividers).
- Not read: Khurana et al. method section (formulas known only through the successor Agro et al.), DeepHit paper text, D-calibration definition, HAWPv2 / v3 pages, ACT paper, the last part of TopoBDA's supplement (Table S6, 8 vs 4 control points), the last 10k characters of PanopticBEV, the GaussianCaR paper itself.
- Not found: polar-ray lane / road methods in BEV; Hermite-spline attention; any combination of survival-style ray verification with top-b hypothesis selection (closest: L-CNN pair-then-verify and Polar R-CNN's top-K of local proposals).

## 6. Search log
Queries (WebSearch):
- Differentiable Raycasting for Self-supervised Occupancy Forecasting Khurana expected depth ray occupancy product (1 - occupancy) method
- DeepHit deep learning approach survival analysis competing risks loss function log-likelihood ranking loss censored AAAI 2018 method
- lane detection lane end point termination prediction where lane ends vertical range start end row-wise "lane ending" CondLaneNet shape head
- line segment detection state of the art 2024 2025 wireframe sAP scalable deep line segment detection ScaleLSD DeepLSD HAWPv3 line distance field
- BEV segmentation ground truth visibility mask occluded regions camera-only nuScenes lidar ray casting visible area map prediction occlusion-aware evaluation
- polar rays lane detection road boundary radial ray regression from ego vehicle BEV polar coordinates lane marking detection ray-based representation
- volume rendering transmittance along rays BEV map segmentation lane road occupancy ray-based BEV mapping alpha compositing camera rays map prediction
- log-polar sampling feature pyramid foveated multi-scale sampling distance-dependent scale selection BEV lane centerline detection far range coarse features
- Hermite spline OR Bezier curve guided deformable attention sampling along centerline lane graph BEV decoder reference points along curve
- survival analysis hazard function applied to line tracing OR lane centerline extension probability "survival" product of continuation probabilities along ray learned stopping criterion curvilinear structure tracing
- learned stopping probability per step curvilinear structure tracing road network tracing "stop" probability trained with censored endpoint cumulative product of continue probabilities
Pages fetched (WebFetch; ar5iv = ar5iv.labs.arxiv.org/html/<id>):
- L-CNN 1905.03246, HAWP 2003.01663, LETR 2101.01909, ELSD 2104.14205 (abstract, then ar5iv), DeepLSD 2212.07766, ScaleLSD arxiv.org/html/2506.09369
- NeRF 2003.08934, Mip-NeRF 360 2111.12077, UniPAD arxiv.org/html/2310.08370, RenderOcc arxiv.org/html/2309.09502, Vampire arxiv.org/html/2312.11837, Agro et al. 2302.13130 (raycasting equations of the Khurana successor)
- Khurana 2210.01917: ar5iv html (failed), arxiv pdf (binary), ECVA pdf (binary), CMU page (redirect), ar5iv abs (failed), alphaxiv overview (abstract only), GitHub README (ok), arXiv abs (ok), deepai page (not the paper)
- Nnet-survival 1805.00917, Kvamme-Borgan 1910.06724, DRSA 1809.02403, DeepHit (AAAI search; par.nsf.gov PDF binary; authors' code raw `class_DeepHit.py`), pycox raw `loss.py`, PonderNet arxiv.org/abs and arxiv.org/html/2107.05407v2, D-calibration arxiv abs 1811.11347
- Cascade R-CNN 1712.00726, Deformable DETR 2010.04159 + raw `ms_deform_attn.py`, `ms_deform_attn_func.py`, `main.py`, DINO 2203.03605, DAB-DETR 2201.12329, Sparse R-CNN 2011.12450, RAFT 2003.12039 + raw `corr.py`, `raft.py`, Sparse4D 2211.10581 and arxiv.org/html/2311.11722, DAT 2201.00520
- CLRNet 2203.10350 + raw `clr_head.py`, LaneATT 2010.12035, Polar R-CNN arxiv.org/html/2411.01499 (offsets 0 and 100000), BezierLaneNet 2203.02431, BezierFormer arxiv.org/html/2404.16304v1, LaneSegNet arxiv.org/html/2312.16108, TopoBDA arxiv.org/html/2412.18951 (offsets 0 and 100000), TopoCurve neurips poster 149504 (abstract), CondLaneNet 2105.05003, UFLD 2004.11757, GANet 2204.07335, Anchor3DLane 2301.02371 (first tried a wrong id 2301.12845), CurveFormer 2209.07989, LATR arxiv.org/html/2308.04583
- PolarMask 1909.13226, PolarFormer 2206.15398, PolarBEVDet arxiv.org/html/2408.16200v3, FPN 1612.03144, Occ3D 2304.14365 (first 100k characters), PanopticBEV 2108.03227 (first 100k characters), parametric-depth BEV arxiv abs 2307.04106
- search: GaussianCaR Gaussian splatting efficient camera-radar fusion BEV map segmentation nuScenes vehicle drivable area outputs
- wasted or wrong: RoadTracer abstract (not needed), HAWPv3 id 2110.08496 (a different paper)
