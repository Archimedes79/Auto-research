# VS01 - Discrete-time hazard network with right censoring (logistic hazard = Nnet-survival)

*Pass: loop 16 (2026-10-23), cluster `verify_sampling` (verification along rays, sampling, survival and hazard read-outs). Reading depth: **M+C (pycox `loss.py` read)** (A abstract, M method, C code page). Verified: checked. Pipeline stage: S3, L.*

## Papers

A scalable discrete-time survival model for neural networks (Gensheimer, Narasimhan, PeerJ 2019, arXiv 2018) https://arxiv.org/abs/1805.00917 ; Continuous and discrete-time survival prediction with neural networks (Kvamme, Borgan, arXiv 2019; journal version from memory) https://arxiv.org/abs/1910.06724 ; pycox implementation https://github.com/havakv/pycox

## Idea

Cut the "time" axis (for us: distance along a ray) into intervals. One sigmoid per interval gives the conditional hazard h_j = P(fail in interval j | alive at its start); survival through j is S_j = prod_{i<=j}(1 - h_i). Training is binary cross-entropy over the intervals up to the observed one; a censored case just stops contributing after its last observed interval. This is exactly the algebra of our survival product (nu_k = 1 - h_k).

## Formulation

event in interval k: log L = log h_k + sum_{j<k} log(1 - h_j); censored: sum of log(1 - h_j) over the survived intervals. The censoring-interval CONVENTION differs between sources: Nnet-survival credits a censored case with the interval of censoring only if the censoring time lies in its second half (labels surv_s / surv_f; loss = sum_i [ln(1 + surv_s(i)(p(i) - 1)) + ln(1 - surv_f(i) p(i))], p = predicted conditional survival); Kvamme-Borgan and pycox `nll_logistic_hazard` count the censoring interval as survived (BCE over j = 0..idx inclusive, target 1 only at the event interval); pycox `nll_pmf` (PMF form) excludes it (mass strictly after); pycox `nll_pc_hazard` (hazard = softplus(phi), piecewise constant) weights the last interval by the observed fraction `interval_frac`. Intervals: 15 - 40 in most Nnet-survival experiments (19 for SUPPORT), spaced wider late so every interval holds a similar number of events; Kvamme-Borgan simulated grids of 5 / 25 / 100 / 250 points, the number of intervals must be tuned (fine grid: less bias, more variance), Kaplan-Meier-quantile grid slightly better than equidistant, constant-density or constant-hazard interpolation inside an interval reduces sensitivity to the grid. Nnet-survival also offers a proportional-hazards variant (one output, 1 - h_i = (1 - h_base)^exp(X beta)).

## Input

point_set of per-interval features (our use: ring samples from bev_raster + context). output: scores (1 - h_j per interval).

## Training

likelihood = masked BCE; labels (interval of event or censoring, event flag); matching_free.

## Differentiable?

yes (sigmoid, sums, products; no argmax).

## Sensors and data

tabular medical data (SUPPORT, METABRIC, FLCHAIN, NWTCO, GBSG), MNIST survival; nothing spatial.

## Reported results

Nnet-survival SUPPORT test (n = 2,732): C-index 0.732, Brier at 6 mo / 1 y / 3 y 0.181 / 0.184 / 0.177 (Cox PH 0.734; 0.183 / 0.186 / 0.178); calibration shown only qualitatively (predicted and Kaplan-Meier curves "superimposed" in the simulation). Kvamme-Borgan 5-fold concordance, Logistic-Hazard: FLCHAIN 0.792, METABRIC 0.658, SUPPORT 0.625 (PMF 0.786 / 0.632 / 0.627); integrated Brier Logistic-Hazard 0.0918 / 0.172 / 0.213; PMF slightly worse and less stable across grids (paper text).

## Compute

negligible (n sigmoids).

## Code

checked https://github.com/havakv/pycox (loss.py); Nnet-survival repo https://github.com/MGensheimer/nnet-survival (URL from memory, not opened)

## Fit to our project

reimplement - S - ring gaps are unequal (4 / 4 / 4 / 6 / 6 m) and the convention at the BEV edge or in occlusion must be fixed (notes 3b).

## Needs

ordered samples per ray; labels for sample k only while all earlier samples were alive (BCE masked after the first failure); a censoring flag per ray (range or occlusion).

## Conflicts

one failure per ray - a ray that leaves the lane and re-enters (occluded stretch, dashed or crossing lanes) violates the single-event model; a marginal "near lane" label at every sample is NOT the conditional hazard.

## Tried in our experiments

none (E94 predicts events continue / end / fork / merge and a step class, not a survival product).
